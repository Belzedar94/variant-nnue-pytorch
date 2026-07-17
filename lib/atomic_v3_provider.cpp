#include "atomic_v3_provider.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
    #include <fcntl.h>
    #include <io.h>
#else
    #include <fcntl.h>
    #include <sys/stat.h>
    #include <unistd.h>
#endif

#include "../external/Atomic-Stockfish/src/atomic_init.h"
#include "../external/Atomic-Stockfish/src/data/atomic_bin_v2_manifest.h"
#include "../external/Atomic-Stockfish/src/data/atomic_bin_v2_wire.h"
#include "../external/Atomic-Stockfish/src/data/sha256.h"
#include "../external/Atomic-Stockfish/src/nnue/atomic_v3/full_refresh.h"

namespace {

namespace AtomicData = Stockfish::Data;
namespace AtomicV3   = Stockfish::Eval::NNUE::AtomicV3;

constexpr std::uint64_t MaximumManifestBytes = 64ULL * 1024ULL * 1024ULL;
constexpr std::uint32_t MaximumBatchSize      = 1U << 20;
constexpr std::uint32_t MaximumManifests      = 100000;
constexpr std::size_t   HashBufferBytes       = 1U << 20;
constexpr std::size_t   RecordBufferCount     = 4096;

thread_local std::array<char, 1024> CreationError{};

void set_creation_error(std::string_view message) noexcept {
    const std::size_t count = std::min(message.size(), CreationError.size() - 1);
    std::memcpy(CreationError.data(), message.data(), count);
    CreationError[count] = '\0';
}

bool lower_sha256(std::string_view value) {
    if (value.size() != 64)
        return false;
    return std::all_of(value.begin(), value.end(), [](char character) {
        return (character >= '0' && character <= '9')
            || (character >= 'a' && character <= 'f');
    });
}

void update_u32(AtomicData::Sha256& hash, std::uint32_t value) {
    std::array<std::uint8_t, 4> bytes{};
    for (unsigned index = 0; index < bytes.size(); ++index)
        bytes[index] = std::uint8_t(value >> (8 * index));
    hash.update(bytes.data(), bytes.size());
}

void update_u64(AtomicData::Sha256& hash, std::uint64_t value) {
    std::array<std::uint8_t, 8> bytes{};
    for (unsigned index = 0; index < bytes.size(); ++index)
        bytes[index] = std::uint8_t(value >> (8 * index));
    hash.update(bytes.data(), bytes.size());
}

std::string system_message(int error) {
    return std::generic_category().message(error ? error : EIO);
}

#ifdef _WIN32
struct FileIdentity {
    DWORD volume = 0;
    DWORD high   = 0;
    DWORD low    = 0;

    bool operator==(const FileIdentity& other) const noexcept {
        return volume == other.volume && high == other.high && low == other.low;
    }
    bool operator!=(const FileIdentity& other) const noexcept { return !(*this == other); }
    bool operator<(const FileIdentity& other) const noexcept {
        if (volume != other.volume)
            return volume < other.volume;
        return high != other.high ? high < other.high : low < other.low;
    }
};

struct ChangeToken {
    FILETIME      creation{};
    FILETIME      write{};
    std::uint64_t size = 0;

    bool operator==(const ChangeToken& other) const noexcept {
        return creation.dwLowDateTime == other.creation.dwLowDateTime
            && creation.dwHighDateTime == other.creation.dwHighDateTime
            && write.dwLowDateTime == other.write.dwLowDateTime
            && write.dwHighDateTime == other.write.dwHighDateTime && size == other.size;
    }
    bool operator!=(const ChangeToken& other) const noexcept { return !(*this == other); }
};

bool inspect_handle(HANDLE handle, FileIdentity& identity, ChangeToken& token) {
    FILE_ATTRIBUTE_TAG_INFO    tag{};
    BY_HANDLE_FILE_INFORMATION info{};
    if (!::GetFileInformationByHandleEx(handle, FileAttributeTagInfo, &tag, sizeof(tag))
        || (tag.FileAttributes & (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DIRECTORY))
        || !::GetFileInformationByHandle(handle, &info))
        return false;
    identity       = {info.dwVolumeSerialNumber, info.nFileIndexHigh, info.nFileIndexLow};
    token.creation = info.ftCreationTime;
    token.write    = info.ftLastWriteTime;
    token.size = (std::uint64_t(info.nFileSizeHigh) << 32) | info.nFileSizeLow;
    return true;
}

int open_regular(const std::filesystem::path& path, FileIdentity& identity, ChangeToken& token) {
    const HANDLE handle =
      ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr,
                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    if (handle == INVALID_HANDLE_VALUE)
    {
        errno = EACCES;
        return -1;
    }
    if (!inspect_handle(handle, identity, token))
    {
        ::CloseHandle(handle);
        errno = EACCES;
        return -1;
    }
    const int descriptor =
      ::_open_osfhandle(reinterpret_cast<std::intptr_t>(handle), _O_RDONLY | _O_BINARY);
    if (descriptor == -1)
        ::CloseHandle(handle);
    return descriptor;
}

bool inspect_descriptor(int descriptor, FileIdentity& identity, ChangeToken& token) {
    const std::intptr_t native = ::_get_osfhandle(descriptor);
    return native != -1 && inspect_handle(reinterpret_cast<HANDLE>(native), identity, token);
}

void close_descriptor(int descriptor) { ::_close(descriptor); }

bool seek_absolute(int descriptor, std::uint64_t offset) {
    if (offset > std::uint64_t(std::numeric_limits<__int64>::max()))
        return false;
    const auto target = static_cast<__int64>(offset);
    return ::_lseeki64(descriptor, target, SEEK_SET) == target;
}

int read_descriptor(int descriptor, void* bytes, std::size_t size) {
    return ::_read(descriptor, bytes,
                   unsigned(std::min<std::size_t>(size, unsigned(-1) >> 1)));
}

int write_descriptor(int descriptor, const void* bytes, std::size_t size) {
    return ::_write(descriptor, bytes,
                    unsigned(std::min<std::size_t>(size, unsigned(-1) >> 1)));
}

using BCryptGenRandomFunction = LONG(WINAPI*)(void*, unsigned char*, ULONG, ULONG);

struct SystemRandomProvider {
    HMODULE                 module    = nullptr;
    BCryptGenRandomFunction generate  = nullptr;
    DWORD                   loadError = ERROR_SUCCESS;

    SystemRandomProvider() {
        module = ::LoadLibraryW(L"bcrypt.dll");
        if (!module)
        {
            loadError = ::GetLastError();
            return;
        }
        const FARPROC address = ::GetProcAddress(module, "BCryptGenRandom");
        if (!address)
        {
            loadError = ::GetLastError();
            return;
        }
        static_assert(sizeof(generate) == sizeof(address));
        std::memcpy(&generate, &address, sizeof(generate));
    }

    ~SystemRandomProvider() {
        if (module)
            ::FreeLibrary(module);
    }

    SystemRandomProvider(const SystemRandomProvider&)            = delete;
    SystemRandomProvider& operator=(const SystemRandomProvider&) = delete;
};

const SystemRandomProvider& system_random_provider() {
    static const SystemRandomProvider provider;
    return provider;
}

int create_private_snapshot() {
    std::error_code       pathError;
    std::filesystem::path directory = std::filesystem::temp_directory_path(pathError);
    if (pathError || directory.empty())
        throw std::invalid_argument("cannot locate the system temporary directory: "
                                    + pathError.message());

    const auto& provider = system_random_provider();
    if (!provider.generate)
        throw std::invalid_argument(
          "cannot load the Windows system random provider: "
          + std::system_category().message(static_cast<int>(provider.loadError)));

    constexpr ULONG   UseSystemPreferredRng = 0x00000002UL;
    constexpr wchar_t Hex[]                 = L"0123456789abcdef";
    for (unsigned attempt = 0; attempt < 128; ++attempt)
    {
        std::array<unsigned char, 32> random{};
        const LONG status = provider.generate(
          nullptr, random.data(), static_cast<ULONG>(random.size()), UseSystemPreferredRng);
        if (status < 0)
            throw std::invalid_argument(
              "Windows system random generation failed while creating a private Atomic V3 snapshot");

        std::wstring name = L"atomic-bin-v2-reader-";
        name.reserve(name.size() + random.size() * 2 + 4);
        for (const unsigned char byte : random)
        {
            name.push_back(Hex[byte >> 4]);
            name.push_back(Hex[byte & 0x0F]);
        }
        name += L".tmp";
        const std::filesystem::path path = directory / name;
        const HANDLE handle = ::CreateFileW(
          path.c_str(), GENERIC_READ | GENERIC_WRITE | DELETE, 0, nullptr, CREATE_NEW,
          FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE | FILE_FLAG_SEQUENTIAL_SCAN,
          nullptr);
        if (handle == INVALID_HANDLE_VALUE)
        {
            const DWORD error = ::GetLastError();
            if (error == ERROR_FILE_EXISTS || error == ERROR_ALREADY_EXISTS)
                continue;
            throw std::invalid_argument(
              "cannot create a private auto-deleting Atomic V3 snapshot: "
              + std::system_category().message(static_cast<int>(error)));
        }

        const int descriptor = ::_open_osfhandle(reinterpret_cast<std::intptr_t>(handle),
                                                  _O_RDWR | _O_BINARY | _O_NOINHERIT);
        if (descriptor < 0)
        {
            const int error = errno;
            ::CloseHandle(handle);
            throw std::invalid_argument(
              "cannot attach a descriptor to the private Atomic V3 snapshot: "
              + system_message(error));
        }
        return descriptor;
    }
    throw std::invalid_argument("cannot allocate a unique private Atomic V3 snapshot name");
}
#else
struct FileIdentity {
    dev_t device = 0;
    ino_t inode  = 0;

    bool operator==(const FileIdentity& other) const noexcept {
        return device == other.device && inode == other.inode;
    }
    bool operator!=(const FileIdentity& other) const noexcept { return !(*this == other); }
    bool operator<(const FileIdentity& other) const noexcept {
        return device != other.device ? device < other.device : inode < other.inode;
    }
};

struct ChangeToken {
    std::int64_t modifiedSeconds     = 0;
    long         modifiedNanoseconds = 0;
    std::int64_t changedSeconds      = 0;
    long         changedNanoseconds  = 0;
    std::uint64_t size               = 0;

    bool operator==(const ChangeToken& other) const noexcept {
        return modifiedSeconds == other.modifiedSeconds
            && modifiedNanoseconds == other.modifiedNanoseconds
            && changedSeconds == other.changedSeconds
            && changedNanoseconds == other.changedNanoseconds && size == other.size;
    }
    bool operator!=(const ChangeToken& other) const noexcept { return !(*this == other); }
};

bool inspect_status(const struct stat& status, FileIdentity& identity, ChangeToken& token) {
    if (!S_ISREG(status.st_mode) || status.st_size < 0)
        return false;
    identity   = {status.st_dev, status.st_ino};
    token.size = std::uint64_t(status.st_size);
    #if defined(__APPLE__)
    token.modifiedSeconds     = status.st_mtimespec.tv_sec;
    token.modifiedNanoseconds = status.st_mtimespec.tv_nsec;
    token.changedSeconds      = status.st_ctimespec.tv_sec;
    token.changedNanoseconds  = status.st_ctimespec.tv_nsec;
    #else
    token.modifiedSeconds     = status.st_mtim.tv_sec;
    token.modifiedNanoseconds = status.st_mtim.tv_nsec;
    token.changedSeconds      = status.st_ctim.tv_sec;
    token.changedNanoseconds  = status.st_ctim.tv_nsec;
    #endif
    return true;
}

int open_regular(const std::filesystem::path& path, FileIdentity& identity, ChangeToken& token) {
    int flags = O_RDONLY;
    #ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
    #endif
    #ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
    #endif
    #ifdef O_NONBLOCK
    flags |= O_NONBLOCK;
    #endif
    int descriptor;
    do
    {
        descriptor = ::open(path.c_str(), flags);
    } while (descriptor == -1 && errno == EINTR);
    if (descriptor == -1)
        return -1;
    struct stat status{};
    if (::fstat(descriptor, &status) != 0 || !inspect_status(status, identity, token))
    {
        ::close(descriptor);
        errno = EINVAL;
        return -1;
    }
    #ifdef O_NONBLOCK
    int descriptorFlags;
    do
    {
        descriptorFlags = ::fcntl(descriptor, F_GETFL);
    } while (descriptorFlags == -1 && errno == EINTR);
    int clearResult = 0;
    if (descriptorFlags != -1 && (descriptorFlags & O_NONBLOCK))
        do
        {
            clearResult = ::fcntl(descriptor, F_SETFL, descriptorFlags & ~O_NONBLOCK);
        } while (clearResult == -1 && errno == EINTR);
    if (descriptorFlags == -1 || clearResult == -1)
    {
        const int error = errno;
        ::close(descriptor);
        errno = error;
        return -1;
    }
    #endif
    return descriptor;
}

bool inspect_descriptor(int descriptor, FileIdentity& identity, ChangeToken& token) {
    struct stat status{};
    return ::fstat(descriptor, &status) == 0 && inspect_status(status, identity, token);
}

void close_descriptor(int descriptor) { ::close(descriptor); }

bool seek_absolute(int descriptor, std::uint64_t offset) {
    if constexpr (sizeof(off_t) <= sizeof(std::uint64_t))
        if (offset > std::uint64_t(std::numeric_limits<off_t>::max()))
            return false;
    return ::lseek(descriptor, off_t(offset), SEEK_SET) == off_t(offset);
}

ssize_t read_descriptor(int descriptor, void* bytes, std::size_t size) {
    return ::read(descriptor, bytes, size);
}

ssize_t write_descriptor(int descriptor, const void* bytes, std::size_t size) {
    return ::write(descriptor, bytes, size);
}

int create_private_snapshot() {
    std::error_code       pathError;
    std::filesystem::path directory = std::filesystem::temp_directory_path(pathError);
    if (pathError || directory.empty())
        throw std::invalid_argument("cannot locate the system temporary directory: "
                                    + pathError.message());
    std::string pattern = (directory / "atomic-bin-v2-reader-XXXXXX").native();
    int descriptor = -1;
#if defined(__linux__) && defined(O_CLOEXEC)
    // Linux exposes the only creation primitive in our supported POSIX matrix
    // which can combine mkstemp's exclusive name allocation with CLOEXEC in
    // the creating syscall.  A later F_SETFD would leave a fork/exec race.
    descriptor = ::mkostemp(pattern.data(), O_CLOEXEC);
#else
    // There is no portable way to add FD_CLOEXEC atomically to mkstemp.  The
    // supported native POSIX target is Linux; other POSIX targets fail closed
    // instead of briefly publishing an inheritable authenticated snapshot.
    throw std::invalid_argument(
      "private Atomic V3 snapshots require atomic mkostemp O_CLOEXEC support");
#endif
    if (descriptor < 0)
        throw std::invalid_argument("cannot create a private Atomic V3 snapshot: "
                                    + system_message(errno));
    if (::unlink(pattern.data()) != 0)
    {
        const int error = errno;
        close_descriptor(descriptor);
        throw std::invalid_argument("cannot unlink the private Atomic V3 snapshot: "
                                    + system_message(error));
    }
    const int flags = ::fcntl(descriptor, F_GETFD);
    if (flags < 0 || !(flags & FD_CLOEXEC))
    {
        const int error = flags < 0 ? errno : EINVAL;
        close_descriptor(descriptor);
        throw std::invalid_argument(
          "private Atomic V3 snapshot was not created non-inheritable: "
          + system_message(error));
    }
    return descriptor;
}
#endif

class OwnedDescriptor {
   public:
    OwnedDescriptor() = default;
    explicit OwnedDescriptor(int value) : descriptor(value) {}
    ~OwnedDescriptor() { reset(); }

    OwnedDescriptor(const OwnedDescriptor&)            = delete;
    OwnedDescriptor& operator=(const OwnedDescriptor&) = delete;

    OwnedDescriptor(OwnedDescriptor&& other) noexcept : descriptor(other.release()) {}
    OwnedDescriptor& operator=(OwnedDescriptor&& other) noexcept {
        if (this != &other)
            reset(other.release());
        return *this;
    }

    int get() const noexcept { return descriptor; }
    explicit operator bool() const noexcept { return descriptor >= 0; }
    int release() noexcept {
        const int result = descriptor;
        descriptor       = -1;
        return result;
    }
    void reset(int value = -1) noexcept {
        if (descriptor >= 0)
            close_descriptor(descriptor);
        descriptor = value;
    }

   private:
    int descriptor = -1;
};

OwnedDescriptor open_regular_owned(const std::filesystem::path& path,
                                   FileIdentity&                identity,
                                   ChangeToken&                 token) {
    errno                = 0;
    const int descriptor = open_regular(path, identity, token);
    if (descriptor < 0)
        throw std::invalid_argument("cannot open regular non-link Atomic V3 shard "
                                    + path.u8string() + ": " + system_message(errno));
    return OwnedDescriptor(descriptor);
}

void read_exact(int descriptor, void* output, std::size_t size, std::string_view context) {
    auto*       bytes  = static_cast<std::uint8_t*>(output);
    std::size_t offset = 0;
    while (offset < size)
    {
        errno            = 0;
        const auto count = read_descriptor(descriptor, bytes + offset, size - offset);
        if (count > 0)
        {
            offset += std::size_t(count);
            continue;
        }
        if (count < 0 && errno == EINTR)
            continue;
        throw std::invalid_argument(std::string(context)
                                    + (count == 0 ? " ended unexpectedly" : " failed: "
                                                         + system_message(errno)));
    }
}

void write_exact(int descriptor, const void* input, std::size_t size) {
    const auto* bytes  = static_cast<const std::uint8_t*>(input);
    std::size_t offset = 0;
    while (offset < size)
    {
        errno            = 0;
        const auto count = write_descriptor(descriptor, bytes + offset, size - offset);
        if (count > 0)
        {
            offset += std::size_t(count);
            continue;
        }
        if (count < 0 && errno == EINTR)
            continue;
        throw std::invalid_argument("private Atomic V3 snapshot write failed: "
                                    + system_message(errno));
    }
}

std::string hash_descriptor(int descriptor, std::uint64_t bytes) {
    if (!seek_absolute(descriptor, 0))
        throw std::invalid_argument("cannot seek Atomic V3 descriptor for SHA-256");
    AtomicData::Sha256       digest;
    std::vector<std::uint8_t> buffer(HashBufferBytes);
    std::uint64_t             remaining = bytes;
    while (remaining != 0)
    {
        const std::size_t count =
          static_cast<std::size_t>(std::min<std::uint64_t>(remaining, buffer.size()));
        read_exact(descriptor, buffer.data(), count, "Atomic V3 SHA-256 read");
        digest.update(buffer.data(), count);
        remaining -= count;
    }
    return digest.hex_digest();
}

struct ManifestSpec {
    std::filesystem::path           path;
    std::string                     payload;
    std::string                     sha256;
    AtomicData::AtomicBinV2Manifest metadata;
    std::vector<FileIdentity>       shardIdentities;
    std::uint64_t                   records = 0;
    std::uint64_t                   firstRecord = 0;
};

AtomicData::AtomicBinV2Manifest load_exact_manifest(const ManifestSpec& spec) {
    AtomicData::AtomicBinV2Manifest trusted{};
    AtomicData::DataResult loaded = AtomicData::load_atomic_bin_v2_manifest(spec.path, trusted);
    if (!loaded)
        throw std::invalid_argument(
          "Atomic V3 manifest changed after authentication: " + loaded.message);
    std::string canonical;
    AtomicData::DataResult rendered =
      AtomicData::render_atomic_bin_v2_manifest(trusted, canonical);
    if (!rendered || canonical != spec.payload)
        throw std::invalid_argument("Atomic V3 manifest changed after authentication");
    AtomicData::Sha256 digest;
    digest.update(canonical);
    if (digest.hex_digest() != spec.sha256)
        throw std::invalid_argument("Atomic V3 manifest SHA-256 differs from provider contract");
    if (trusted.records != spec.records)
        throw std::invalid_argument("Atomic V3 manifest record count differs from provider contract");
    return trusted;
}

class AuthenticatedShard {
   public:
    AuthenticatedShard() = default;
    ~AuthenticatedShard() { close(); }

    AuthenticatedShard(const AuthenticatedShard&)            = delete;
    AuthenticatedShard& operator=(const AuthenticatedShard&) = delete;

    void stage(const AtomicData::AtomicBinV2ManifestShard& expected,
               const FileIdentity&                          expectedSourceIdentity,
               std::uint64_t                                recordIndex) {
        close();
        if (recordIndex > expected.records)
            throw std::invalid_argument("Atomic V3 shard seek is outside its record count");
        if (expected.bytes > AtomicData::Sha256MaxByteCount)
            throw std::invalid_argument("Atomic V3 shard exceeds the SHA-256 byte domain");
        std::uint64_t canonicalBytes = 0;
        const AtomicData::DataResult sized =
          AtomicData::atomic_bin_v2_file_size(expected.records, canonicalBytes);
        if (!sized || canonicalBytes != expected.bytes)
            throw std::invalid_argument("Atomic V3 shard size is not canonical");

        FileIdentity sourceIdentity{};
        ChangeToken  sourceToken{};
        OwnedDescriptor source = open_regular_owned(expected.path, sourceIdentity, sourceToken);
        if (sourceIdentity != expectedSourceIdentity)
            throw std::invalid_argument(
              "Atomic V3 shard pathname identity changed after provider authentication");
        if (sourceToken.size != expected.bytes)
            throw std::invalid_argument("Atomic V3 shard byte count differs from manifest");

        OwnedDescriptor snapshot(create_private_snapshot());
        AtomicData::Sha256 digest;
        std::vector<std::uint8_t> stagingBuffer(HashBufferBytes);
        std::uint64_t remaining = expected.bytes;
        while (remaining != 0)
        {
            const std::size_t count = static_cast<std::size_t>(
              std::min<std::uint64_t>(remaining, stagingBuffer.size()));
            read_exact(source.get(), stagingBuffer.data(), count,
                       "Atomic V3 source staging read");
            write_exact(snapshot.get(), stagingBuffer.data(), count);
            digest.update(stagingBuffer.data(), count);
            remaining -= count;
        }

        FileIdentity sourceIdentityAfter{};
        ChangeToken  sourceTokenAfter{};
        if (!inspect_descriptor(source.get(), sourceIdentityAfter, sourceTokenAfter)
            || sourceIdentityAfter != sourceIdentity || sourceTokenAfter != sourceToken)
            throw std::invalid_argument("Atomic V3 shard changed while being staged");

        FileIdentity pathIdentity{};
        ChangeToken  pathToken{};
        OwnedDescriptor pathCheck = open_regular_owned(expected.path, pathIdentity, pathToken);
        if (pathIdentity != sourceIdentity || pathToken != sourceToken)
            throw std::invalid_argument("Atomic V3 shard pathname changed while being staged");
        pathCheck.reset();
        source.reset();

        if (digest.hex_digest() != expected.sha256)
            throw std::invalid_argument("Atomic V3 shard SHA-256 differs from manifest");

        FileIdentity privateIdentity{};
        ChangeToken  privateToken{};
        if (!inspect_descriptor(snapshot.get(), privateIdentity, privateToken)
            || privateToken.size != expected.bytes)
            throw std::invalid_argument(
              "private Atomic V3 shard snapshot size or identity is invalid");
        const std::string snapshotHash = hash_descriptor(snapshot.get(), expected.bytes);
        FileIdentity      privateIdentityAfter{};
        ChangeToken       privateTokenAfter{};
        if (snapshotHash != expected.sha256
            || !inspect_descriptor(snapshot.get(), privateIdentityAfter, privateTokenAfter)
            || privateIdentityAfter != privateIdentity || privateTokenAfter != privateToken)
            throw std::invalid_argument(
              "private Atomic V3 shard snapshot failed full SHA-256 authentication");

        if (!seek_absolute(snapshot.get(), 0))
            throw std::invalid_argument("cannot seek private Atomic V3 shard header");
        AtomicData::AtomicBinV2Header header{};
        read_exact(snapshot.get(), header.data(), header.size(),
                   "private Atomic V3 shard header read");
        std::uint64_t headerRecords = 0;
        const AtomicData::DataResult decoded =
          AtomicData::decode_atomic_bin_v2_header(header, headerRecords);
        if (!decoded || headerRecords != expected.records)
            throw std::invalid_argument("private Atomic V3 shard header differs from manifest");
        FileIdentity headerIdentity{};
        ChangeToken  headerToken{};
        if (!inspect_descriptor(snapshot.get(), headerIdentity, headerToken)
            || headerIdentity != privateIdentity || headerToken != privateToken)
            throw std::invalid_argument(
              "private Atomic V3 shard snapshot changed during header validation");

        const std::uint64_t offset = AtomicData::AtomicBinV2HeaderSize
                                   + recordIndex * AtomicData::AtomicBinV2RecordSize;
        if (!seek_absolute(snapshot.get(), offset))
            throw std::invalid_argument("cannot seek authenticated Atomic V3 shard snapshot");

        descriptor       = snapshot.release();
        snapshotIdentity = privateIdentity;
        snapshotToken    = privateToken;
        recordCount      = expected.records;
        nextRecord       = recordIndex;
        bufferIndex = bufferCount = 0;
    }

    void seek(std::uint64_t recordIndex) {
        if (descriptor < 0 || recordIndex > recordCount)
            throw std::invalid_argument("Atomic V3 shard seek is outside its record count");
        const std::uint64_t offset = AtomicData::AtomicBinV2HeaderSize
                                   + recordIndex * AtomicData::AtomicBinV2RecordSize;
        verify_snapshot();
        if (!seek_absolute(descriptor, offset))
            throw std::invalid_argument("cannot seek authenticated Atomic V3 shard");
        verify_snapshot();
        nextRecord = recordIndex;
        bufferIndex = bufferCount = 0;
    }

    bool next(AtomicData::AtomicBinV2Record& output) {
        if (descriptor < 0)
            throw std::logic_error("Atomic V3 shard snapshot is not active");
        if (nextRecord == recordCount)
            return false;
        if (bufferIndex == bufferCount)
            refill();
        output = recordBuffer[bufferIndex++];
        ++nextRecord;
        return true;
    }

    bool exhausted() const noexcept {
        return descriptor >= 0 && nextRecord == recordCount;
    }

    void close() noexcept {
        if (descriptor >= 0)
            close_descriptor(descriptor);
        descriptor = -1;
        snapshotIdentity = {};
        snapshotToken = {};
        recordCount = nextRecord = 0;
        bufferIndex = bufferCount = 0;
    }

   private:
    void verify_snapshot() const {
        FileIdentity identity{};
        ChangeToken  token{};
        if (descriptor < 0 || !inspect_descriptor(descriptor, identity, token)
            || identity != snapshotIdentity || token != snapshotToken)
            throw std::invalid_argument("private authenticated Atomic V3 shard snapshot changed");
    }

    void refill() {
        verify_snapshot();
        const std::uint64_t remaining = recordCount - nextRecord;
        bufferCount = static_cast<std::size_t>(
          std::min<std::uint64_t>(remaining, RecordBufferCount));
        bufferIndex = 0;
        const std::size_t bytes = bufferCount * sizeof(AtomicData::AtomicBinV2Record);
        read_exact(descriptor, recordBuffer.data(), bytes,
                   "private Atomic V3 buffered record read");
        verify_snapshot();
    }

    std::array<AtomicData::AtomicBinV2Record, RecordBufferCount> recordBuffer{};
    FileIdentity snapshotIdentity{};
    ChangeToken snapshotToken{};
    std::uint64_t recordCount = 0;
    std::uint64_t nextRecord  = 0;
    std::size_t bufferIndex   = 0;
    std::size_t bufferCount   = 0;
    int descriptor = -1;
};

class FastManifestReader {
   public:
    explicit FastManifestReader(const ManifestSpec& spec) :
        manifest(&spec.metadata), identities(&spec.shardIdentities), records(spec.records) {
        std::uint64_t first = 0;
        firstRecords.reserve(manifest->shards.size());
        for (const auto& shard : manifest->shards)
        {
            firstRecords.push_back(first);
            if (first > std::numeric_limits<std::uint64_t>::max() - shard.records)
                throw std::overflow_error("Atomic V3 manifest shard records overflow uint64");
            first += shard.records;
        }
        if (manifest->shards.empty() || identities->size() != manifest->shards.size()
            || first != records)
            throw std::invalid_argument("Atomic V3 manifest shard totals differ");
    }

    void seek(std::uint64_t recordIndex, AuthenticatedShard& active) {
        if (recordIndex >= records)
            throw std::invalid_argument("Atomic V3 manifest seek is outside its record count");
        const auto upper = std::upper_bound(firstRecords.begin(), firstRecords.end(), recordIndex);
        currentShard = static_cast<std::size_t>(upper - firstRecords.begin() - 1);
        currentRecord = recordIndex;
        position_current(active);
        currentShardPositioned = true;
    }

    bool next(AtomicData::AtomicBinV2Record& output, AuthenticatedShard& active) {
        if (currentRecord == records)
            return false;
        if (!currentShardPositioned)
        {
            position_current(active);
            currentShardPositioned = true;
        }
        if (!active.next(output))
            throw std::logic_error("authenticated Atomic V3 shard ended before its manifest count");
        ++currentRecord;
        if (active.exhausted())
        {
            active.close();
            if (currentRecord != records)
                ++currentShard;
            currentShardPositioned = false;
        }
        return true;
    }

    void close(AuthenticatedShard& active) noexcept {
        active.close();
        currentShardPositioned = false;
    }

   private:
    void position_current(AuthenticatedShard& active) {
        active.stage(manifest->shards[currentShard], (*identities)[currentShard],
                     currentRecord - firstRecords[currentShard]);
    }

    const AtomicData::AtomicBinV2Manifest* manifest = nullptr;
    const std::vector<FileIdentity>* identities = nullptr;
    std::vector<std::uint64_t> firstRecords;
    std::uint64_t records = 0;
    std::uint64_t currentRecord = 0;
    std::size_t currentShard = 0;
    bool currentShardPositioned = false;
};

static_assert(sizeof(FastManifestReader) < 256,
              "FastManifestReader must not embed per-shard record buffers or handles");

std::uint64_t splitmix64(std::uint64_t value) noexcept {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

bool keep_record(std::uint64_t seed,
                 std::uint64_t epoch,
                 std::uint64_t roleRecord,
                 std::uint32_t randomFenSkipping) noexcept {
    if (randomFenSkipping == 0)
        return true;
    const std::uint64_t bound = std::uint64_t(randomFenSkipping) + 1;
    const std::uint64_t threshold = (0ULL - bound) % bound;
    std::uint64_t lane = 0;
    for (;;)
    {
        const std::uint64_t counter = seed
                                    ^ (epoch * 0xD1B54A32D192ED03ULL)
                                    ^ (roleRecord * 0x94D049BB133111EBULL)
                                    ^ (lane++ * 0x9E3779B97F4A7C15ULL);
        const std::uint64_t value = splitmix64(counter);
        if (value >= threshold)
            return value % bound == 0;
    }
}

Stockfish::Piece piece_from_wire(Stockfish::u8 value) {
    using namespace AtomicData;
    switch (value)
    {
    case ATOMIC_BIN_V2_EMPTY :
        return Stockfish::NO_PIECE;
    case ATOMIC_BIN_V2_WHITE_PAWN :
        return Stockfish::W_PAWN;
    case ATOMIC_BIN_V2_WHITE_KNIGHT :
        return Stockfish::W_KNIGHT;
    case ATOMIC_BIN_V2_WHITE_BISHOP :
        return Stockfish::W_BISHOP;
    case ATOMIC_BIN_V2_WHITE_ROOK :
        return Stockfish::W_ROOK;
    case ATOMIC_BIN_V2_WHITE_QUEEN :
        return Stockfish::W_QUEEN;
    case ATOMIC_BIN_V2_WHITE_KING :
        return Stockfish::W_KING;
    case ATOMIC_BIN_V2_BLACK_PAWN :
        return Stockfish::B_PAWN;
    case ATOMIC_BIN_V2_BLACK_KNIGHT :
        return Stockfish::B_KNIGHT;
    case ATOMIC_BIN_V2_BLACK_BISHOP :
        return Stockfish::B_BISHOP;
    case ATOMIC_BIN_V2_BLACK_ROOK :
        return Stockfish::B_ROOK;
    case ATOMIC_BIN_V2_BLACK_QUEEN :
        return Stockfish::B_QUEEN;
    case ATOMIC_BIN_V2_BLACK_KING :
        return Stockfish::B_KING;
    default :
        throw std::invalid_argument("Atomic V3 record contains an unknown piece code");
    }
}

struct FeatureRange {
    std::size_t   offset = 0;
    std::uint32_t size   = 0;
};

struct PerspectiveRows {
    std::int64_t ownKing = 0;
    FeatureRange hm;
    FeatureRange capturePair;
    FeatureRange kingBlastEp;
    FeatureRange blastRing;
};

struct PackedSample {
    float         sideToMoveWhite = 0.0F;
    std::int64_t  pieceCount = 0;
    PerspectiveRows white;
    PerspectiveRows black;
    float         outcome = 0.0F;
    float         score = 0.0F;
    std::int64_t  bucket = 0;
};

static_assert(std::is_trivially_copyable_v<PackedSample>,
              "PackedSample must remain allocation-free");

template<typename IndexAt>
FeatureRange append_feature_range(std::vector<std::int32_t>& arena,
                                  std::uint32_t              count,
                                  IndexAt                    indexAt) {
    if (count > std::numeric_limits<std::size_t>::max() - arena.size())
        throw std::length_error("Atomic V3 feature-index arena overflow");
    const FeatureRange range{arena.size(), count};
    arena.resize(arena.size() + count);
    for (std::uint32_t index = 0; index < count; ++index)
        arena[range.offset + index] = std::int32_t(indexAt(index));
    return range;
}

PerspectiveRows rows_from_emission(const AtomicV3::FullRefreshEmission& emission,
                                   std::vector<std::int32_t>&            arena) {
    PerspectiveRows rows;
    rows.ownKing = std::int64_t(emission.hm.orientation.ownKing);
    rows.hm = append_feature_range(arena, emission.hm.size, [&](std::uint32_t index) {
        return emission.hm.features[index].trainingIndex;
    });
    rows.capturePair =
      append_feature_range(arena, emission.capturePairs.size, [&](std::uint32_t index) {
          return emission.capturePairs.features[index].localIndex;
      });
    rows.kingBlastEp =
      append_feature_range(arena, emission.kingBlastEp.size, [&](std::uint32_t index) {
          return emission.kingBlastEp.features[index].localIndex;
      });
    rows.blastRing =
      append_feature_range(arena, emission.blastRing.size, [&](std::uint32_t index) {
          return emission.blastRing.features[index].localIndex;
      });
    const auto sorted = [&](const FeatureRange& range) {
        const auto begin = arena.begin() + std::ptrdiff_t(range.offset);
        return std::is_sorted(begin, begin + range.size);
    };
    if (!sorted(rows.capturePair) || !sorted(rows.kingBlastEp) || !sorted(rows.blastRing))
        throw std::logic_error("Atomic V3 relation oracle returned noncanonical order");
    return rows;
}

PackedSample pack_sample(const AtomicData::AtomicBinV2RecordFields& fields,
                         std::vector<std::int32_t>&                  arena) {
    AtomicV3::CapturePairSnapshot snapshot{};
    snapshot.sideToMove = fields.position.sideToMove == AtomicData::ATOMIC_BIN_V2_WHITE_TO_MOVE
                            ? Stockfish::WHITE
                            : Stockfish::BLACK;
    snapshot.epSquare = fields.position.enPassantSquare == AtomicData::AtomicBinV2NoSquare
                        ? Stockfish::SQ_NONE
                        : Stockfish::Square(fields.position.enPassantSquare);
    for (std::size_t square = 0; square < fields.position.board.size(); ++square)
        snapshot.board[square] = piece_from_wire(fields.position.board[square]);

    AtomicV3::FullRefreshEmission whiteEmission{};
    AtomicV3::FullRefreshEmission blackEmission{};
    const auto whiteError = AtomicV3::emit_full_refresh(snapshot, Stockfish::WHITE, whiteEmission);
    const auto blackError = AtomicV3::emit_full_refresh(snapshot, Stockfish::BLACK, blackEmission);
    if (whiteError != AtomicV3::CapturePairError::None
        || blackError != AtomicV3::CapturePairError::None)
        throw std::invalid_argument(
          std::string("Atomic V3 full refresh rejected record: white=")
          + AtomicV3::full_refresh_error_message(whiteError) + " black="
          + AtomicV3::full_refresh_error_message(blackError));
    if (whiteEmission.hm.size != blackEmission.hm.size
        || whiteEmission.hm.networkBucket != blackEmission.hm.networkBucket)
        throw std::logic_error("Atomic V3 perspectives disagree on piece count or network bucket");

    const float score = static_cast<float>(fields.score);
    if (static_cast<std::int64_t>(score) != std::int64_t(fields.score))
        throw std::invalid_argument("Atomic V3 score is not exactly representable as float32");

    PackedSample sample;
    sample.sideToMoveWhite = snapshot.sideToMove == Stockfish::WHITE ? 1.0F : 0.0F;
    sample.pieceCount = std::int64_t(whiteEmission.hm.size);
    sample.white       = rows_from_emission(whiteEmission, arena);
    sample.black       = rows_from_emission(blackEmission, arena);
    sample.outcome     = (float(fields.result) + 1.0F) / 2.0F;
    sample.score       = score;
    sample.bucket      = std::int64_t(whiteEmission.hm.networkBucket);
    return sample;
}

struct SparseStorage {
    std::uint32_t            width = 1;
    std::vector<std::int32_t> indices;
    std::vector<float>        values;
};

struct PerspectiveStorage {
    std::vector<std::int64_t> ownKingSquares;
    SparseStorage             hm;
    SparseStorage             capturePair;
    SparseStorage             kingBlastEp;
    SparseStorage             blastRing;
};

template<typename Selector>
SparseStorage pack_sparse(const std::vector<PackedSample>& samples,
                          const std::vector<std::int32_t>& featureIndices,
                          Selector                         selector) {
    SparseStorage storage;
    for (const PackedSample& sample : samples)
    {
        const FeatureRange range = selector(sample);
        if (range.offset > featureIndices.size()
            || range.size > featureIndices.size() - range.offset)
            throw std::logic_error("Atomic V3 feature-index range escaped its arena");
        storage.width = std::max<std::uint32_t>(
          storage.width, range.size);
    }
    if (samples.size() > std::numeric_limits<std::size_t>::max() / storage.width)
        throw std::length_error("Atomic V3 sparse batch size overflow");
    const std::size_t cells = samples.size() * storage.width;
    storage.indices.assign(cells, -1);
    storage.values.assign(cells, 0.0F);
    for (std::size_t row = 0; row < samples.size(); ++row)
    {
        const FeatureRange range = selector(samples[row]);
        for (std::uint32_t column = 0; column < range.size; ++column)
        {
            const std::size_t offset = row * storage.width + column;
            storage.indices[offset] = featureIndices[range.offset + column];
            storage.values[offset]  = 1.0F;
        }
    }
    return storage;
}

PerspectiveStorage pack_perspective(const std::vector<PackedSample>& samples,
                                    const std::vector<std::int32_t>& featureIndices,
                                    bool                             white) {
    PerspectiveStorage storage;
    storage.ownKingSquares.reserve(samples.size());
    for (const PackedSample& sample : samples)
        storage.ownKingSquares.push_back(white ? sample.white.ownKing : sample.black.ownKing);
    const auto perspective = [white](const PackedSample& sample) -> const PerspectiveRows& {
        return white ? sample.white : sample.black;
    };
    storage.hm = pack_sparse(samples, featureIndices, [&](const PackedSample& sample) {
        return perspective(sample).hm;
    });
    storage.capturePair =
      pack_sparse(samples, featureIndices, [&](const PackedSample& sample) {
          return perspective(sample).capturePair;
      });
    storage.kingBlastEp =
      pack_sparse(samples, featureIndices, [&](const PackedSample& sample) {
          return perspective(sample).kingBlastEp;
      });
    storage.blastRing = pack_sparse(samples, featureIndices, [&](const PackedSample& sample) {
        return perspective(sample).blastRing;
    });
    return storage;
}

struct BatchImpl {
    std::vector<float>        sideToMoveWhite;
    std::vector<std::int64_t> pieceCounts;
    PerspectiveStorage        white;
    PerspectiveStorage        black;
    std::vector<float>        outcome;
    std::vector<float>        score;
    std::vector<std::int64_t> bucketIndices;
    AtomicV3ProviderCursorV1  cursorAfter{};

    BatchImpl(const std::vector<PackedSample>& samples,
              const std::vector<std::int32_t>& featureIndices,
              const AtomicV3ProviderCursorV1&  cursor) :
        white(pack_perspective(samples, featureIndices, true)),
        black(pack_perspective(samples, featureIndices, false)),
        cursorAfter(cursor) {
        sideToMoveWhite.reserve(samples.size());
        pieceCounts.reserve(samples.size());
        outcome.reserve(samples.size());
        score.reserve(samples.size());
        bucketIndices.reserve(samples.size());
        for (const PackedSample& sample : samples)
        {
            sideToMoveWhite.push_back(sample.sideToMoveWhite);
            pieceCounts.push_back(sample.pieceCount);
            outcome.push_back(sample.outcome);
            score.push_back(sample.score);
            bucketIndices.push_back(sample.bucket);
        }
    }

    static AtomicV3SparseSliceViewV1 view(const SparseStorage& storage) {
        return {storage.indices.data(), storage.values.data(), storage.width, 0};
    }

    static AtomicV3PerspectiveViewV1 view(const PerspectiveStorage& storage) {
        return {storage.ownKingSquares.data(), view(storage.hm), view(storage.capturePair),
                view(storage.kingBlastEp), view(storage.blastRing)};
    }

    AtomicV3BatchViewV1 view() const {
        AtomicV3BatchViewV1 result{};
        result.abiVersion      = AtomicV3ProviderAbiVersion;
        result.structSize      = sizeof(result);
        result.size            = std::uint32_t(sideToMoveWhite.size());
        result.sideToMoveWhite = sideToMoveWhite.data();
        result.pieceCounts     = pieceCounts.data();
        result.white           = view(white);
        result.black           = view(black);
        result.outcome         = outcome.data();
        result.score           = score.data();
        result.bucketIndices   = bucketIndices.data();
        result.cursorAfter     = cursorAfter;
        return result;
    }
};

class StreamImpl {
   public:
    explicit StreamImpl(const AtomicV3ProviderConfigV1& config) :
        batchSize(config.batchSize),
        randomFenSkipping(config.randomFenSkipping),
        seed(config.seed),
        cyclic(config.cyclic != 0) {
        Stockfish::initialize_atomic_core();
        load_manifests(config);
        binding = compute_binding();
        initialize_cursor(config.resumeCursor);
    }

    std::unique_ptr<BatchImpl> next() {
        if (poisoned)
            throw std::runtime_error("Atomic V3 provider is poisoned after an earlier error");
        if (working.eof)
            return nullptr;

        std::vector<PackedSample> samples;
        samples.reserve(batchSize);
        std::vector<std::int32_t> featureIndices;
        constexpr std::size_t InitialFeaturesPerSample = 256;
        constexpr std::size_t MaximumInitialFeatures   = 4U * 1024U * 1024U;
        const std::size_t initialSamples = std::min<std::size_t>(
          batchSize, MaximumInitialFeatures / InitialFeaturesPerSample);
        featureIndices.reserve(initialSamples * InitialFeaturesPerSample);
        while (samples.size() < batchSize)
        {
            AtomicData::AtomicBinV2Record wire{};
            std::uint64_t sourceEpoch = 0;
            std::uint64_t sourceRoleRecord = 0;
            std::size_t manifestIndex = 0;
            std::uint64_t localRecord = 0;
            if (!next_raw(wire, sourceEpoch, sourceRoleRecord, manifestIndex, localRecord))
                break;
            if (!keep_record(seed, sourceEpoch, sourceRoleRecord, randomFenSkipping))
                continue;
            if (working.acceptedSamples == std::numeric_limits<std::uint64_t>::max())
                throw std::overflow_error("Atomic V3 accepted-sample counter overflow");
            if (working.nextBatchSequence == std::numeric_limits<std::uint64_t>::max())
                throw std::overflow_error("Atomic V3 batch-sequence counter overflow");
            samples.push_back(
              pack_sample(decode_retained(wire, manifestIndex, localRecord), featureIndices));
            ++working.acceptedSamples;
        }
        if (samples.empty())
            return nullptr;
        ++working.nextBatchSequence;
        lastDelivered = working;
        hasDelivered  = true;
        return std::make_unique<BatchImpl>(samples, featureIndices, working);
    }

    void commit() {
        if (!hasDelivered)
            throw std::logic_error("Atomic V3 provider has no delivered batch to commit");
        committed = lastDelivered;
        hasDelivered = false;
    }

    const AtomicV3ProviderCursorV1& committed_cursor() const noexcept { return committed; }

    void set_error(std::string message) noexcept {
        error = std::move(message);
        poisoned = true;
        activeShard.close();
        readers.clear();
        readerPositioned = false;
    }
    const std::string& last_error() const noexcept { return error; }

   private:
    void load_manifests(const AtomicV3ProviderConfigV1& config) {
        if (config.manifestCount == 0 || config.manifestCount > MaximumManifests)
            throw std::invalid_argument("Atomic V3 provider manifest count is outside bounds");
        if (config.manifests == nullptr)
            throw std::invalid_argument("Atomic V3 provider manifest array is null");
        specs.reserve(config.manifestCount);
        std::set<std::filesystem::path> shardPaths;
        std::set<FileIdentity>          shardIdentities;
        std::uint64_t first = 0;
        for (std::uint32_t index = 0; index < config.manifestCount; ++index)
        {
            const AtomicV3ManifestInputV1& input = config.manifests[index];
            if (input.pathUtf8 == nullptr || input.payload == nullptr || input.sha256Hex == nullptr
                || input.pathBytes == 0 || input.pathBytes > 32768
                || input.payloadBytes == 0 || input.payloadBytes > MaximumManifestBytes
                || input.sha256Bytes != 64
                || input.expectedRecords == 0)
                throw std::invalid_argument("Atomic V3 manifest descriptor is invalid");
            const std::string pathBytes(input.pathUtf8, std::size_t(input.pathBytes));
            if (pathBytes.find('\0') != std::string::npos)
                throw std::invalid_argument("Atomic V3 manifest path contains NUL");
            const std::string expectedHash(input.sha256Hex, std::size_t(input.sha256Bytes));
            if (!lower_sha256(expectedHash))
                throw std::invalid_argument("Atomic V3 manifest SHA-256 is not lowercase hex");
            const std::string payload(reinterpret_cast<const char*>(input.payload),
                                      std::size_t(input.payloadBytes));
            AtomicData::Sha256 payloadHash;
            payloadHash.update(payload);
            if (payloadHash.hex_digest() != expectedHash)
                throw std::invalid_argument("Atomic V3 supplied manifest bytes fail SHA-256");

            ManifestSpec spec;
            spec.path        = std::filesystem::u8path(pathBytes);
            spec.payload     = payload;
            spec.sha256      = expectedHash;
            spec.records     = input.expectedRecords;
            spec.firstRecord = first;
            if (first > std::numeric_limits<std::uint64_t>::max() - spec.records)
                throw std::overflow_error("Atomic V3 aggregate record count overflows uint64");
            first += spec.records;

            // Authenticate the named canonical manifest and establish every
            // source identity without retaining a source handle.  Actual
            // shard bytes are staged lazily into one private snapshot.
            spec.metadata = load_exact_manifest(spec);
            spec.shardIdentities.reserve(spec.metadata.shards.size());
            for (const auto& shard : spec.metadata.shards)
            {
                const auto normalized = shard.path.lexically_normal();
                if (!shardPaths.insert(normalized).second)
                    throw std::invalid_argument(
                      "Atomic V3 ordered manifests repeat a shard pathname");
                if (shard.bytes > AtomicData::Sha256MaxByteCount)
                    throw std::invalid_argument(
                      "Atomic V3 shard exceeds the SHA-256 byte domain");
                FileIdentity identity{};
                ChangeToken  token{};
                OwnedDescriptor descriptor = open_regular_owned(shard.path, identity, token);
                if (token.size != shard.bytes)
                    throw std::invalid_argument(
                      "Atomic V3 shard byte count differs from its manifest");
                if (!shardIdentities.insert(identity).second)
                    throw std::invalid_argument(
                      "Atomic V3 ordered manifests repeat a shard filesystem identity");
                spec.shardIdentities.push_back(identity);
            }
            specs.push_back(std::move(spec));
        }
        totalRecords = first;
        readers.resize(specs.size());
    }

    std::array<std::uint8_t, 32> compute_binding() const {
        AtomicData::Sha256 hash;
        // Include the terminating NUL without relying on a hand-counted byte
        // literal. This is part of the persisted cursor binding.
        static constexpr char Domain[] = "atomic-v3-sequential-provider-v1";
        hash.update(Domain, sizeof(Domain));
        update_u32(hash, std::uint32_t(specs.size()));
        for (const ManifestSpec& spec : specs)
        {
            hash.update(spec.sha256);
            update_u64(hash, spec.records);
        }
        update_u32(hash, batchSize);
        update_u32(hash, randomFenSkipping);
        update_u64(hash, seed);
        update_u32(hash, cyclic ? 1U : 0U);
        return hash.digest();
    }

    AtomicV3ProviderCursorV1 initial_cursor() const {
        AtomicV3ProviderCursorV1 cursor{};
        cursor.abiVersion = AtomicV3ProviderAbiVersion;
        cursor.structSize = sizeof(cursor);
        std::copy(binding.begin(), binding.end(), cursor.bindingSha256);
        return cursor;
    }

    void initialize_cursor(const AtomicV3ProviderCursorV1* resume) {
        working = initial_cursor();
        if (resume != nullptr)
        {
            if (resume->abiVersion != AtomicV3ProviderAbiVersion
                || resume->structSize != sizeof(AtomicV3ProviderCursorV1)
                || resume->eof > 1
                || resume->reserved0 != 0
                || std::any_of(std::begin(resume->reserved1), std::end(resume->reserved1),
                               [](std::uint8_t value) { return value != 0; })
                || !std::equal(binding.begin(), binding.end(), resume->bindingSha256))
                throw std::invalid_argument("Atomic V3 resume cursor contract/binding mismatch");
            working = *resume;
        }
        if (!cyclic && working.epoch != 0)
            throw std::invalid_argument("Atomic V3 non-cyclic resume epoch must be zero");
        if (working.eof)
        {
            if (cyclic || working.manifestIndex != specs.size() || working.recordIndex != 0)
                throw std::invalid_argument("Atomic V3 resume EOF cursor is not canonical");
        }
        else
        {
            if (working.manifestIndex >= specs.size()
                || working.recordIndex >= specs[working.manifestIndex].records)
                throw std::invalid_argument("Atomic V3 resume cursor is outside ordered manifests");
            if (working.recordIndex != 0)
                open_current();
        }
        committed = working;
    }

    void open_current() {
        if (readerPositioned)
            return;
        auto& reader = readers[working.manifestIndex];
        if (!reader)
            reader = std::make_unique<FastManifestReader>(specs[working.manifestIndex]);
        reader->seek(working.recordIndex, activeShard);
        readerPositioned = true;
    }

    void finish_manifest() {
        if (readers[working.manifestIndex])
            readers[working.manifestIndex]->close(activeShard);
        else
            activeShard.close();
        readerPositioned = false;
        working.recordIndex = 0;
        ++working.manifestIndex;
        if (working.manifestIndex == specs.size())
        {
            if (cyclic)
            {
                working.manifestIndex = 0;
                if (working.epoch == std::numeric_limits<std::uint64_t>::max())
                    throw std::overflow_error("Atomic V3 provider epoch overflow");
                ++working.epoch;
            }
            else
            {
                working.eof = 1;
            }
        }
    }

    bool next_raw(AtomicData::AtomicBinV2Record& wire,
                  std::uint64_t& sourceEpoch,
                  std::uint64_t& sourceRoleRecord,
                  std::size_t& manifestIndex,
                  std::uint64_t& localRecord) {
        if (working.eof)
            return false;
        open_current();
        const ManifestSpec& spec = specs[working.manifestIndex];
        const std::uint64_t local = working.recordIndex;
        if (!readers[working.manifestIndex]->next(wire, activeShard))
            throw std::logic_error("fast reader ended before the Atomic V3 manifest record count");
        sourceEpoch      = working.epoch;
        sourceRoleRecord = spec.firstRecord + local;
        manifestIndex    = working.manifestIndex;
        localRecord      = local;
        ++working.recordIndex;
        if (working.recordIndex == spec.records)
            finish_manifest();
        return true;
    }

    AtomicData::AtomicBinV2RecordFields decode_retained(
      const AtomicData::AtomicBinV2Record& wire,
      std::size_t manifestIndex,
      std::uint64_t localRecord) const {
        AtomicData::AtomicBinV2RecordFields fields{};
        const AtomicData::DataResult decoded =
          AtomicData::decode_atomic_bin_v2_record_structural(wire, fields);
        if (!decoded)
            throw std::invalid_argument(
              "retained Atomic V3 record is structurally invalid at manifest "
              + std::to_string(manifestIndex) + " record " + std::to_string(localRecord)
              + ": " + decoded.message);
        if (bool(fields.flags & AtomicData::ATOMIC_BIN_V2_ATOMIC960)
            != specs[manifestIndex].metadata.atomic960)
            throw std::invalid_argument(
              "retained Atomic V3 record Atomic960 flag differs from its manifest");
        return fields;
    }

    std::vector<ManifestSpec> specs;
    std::vector<std::unique_ptr<FastManifestReader>> readers;
    AuthenticatedShard activeShard;
    std::array<std::uint8_t, 32> binding{};
    std::uint64_t totalRecords = 0;
    std::uint32_t batchSize = 0;
    std::uint32_t randomFenSkipping = 0;
    std::uint64_t seed = 0;
    bool cyclic = false;
    bool poisoned = false;
    bool hasDelivered = false;
    bool readerPositioned = false;
    std::string error;
    AtomicV3ProviderCursorV1 working{};
    AtomicV3ProviderCursorV1 committed{};
    AtomicV3ProviderCursorV1 lastDelivered{};
};

}  // namespace

struct AtomicV3ProviderStreamV1 {
    explicit AtomicV3ProviderStreamV1(const AtomicV3ProviderConfigV1& config) : impl(config) {}
    StreamImpl impl;
};

struct AtomicV3ProviderBatchV1 {
    explicit AtomicV3ProviderBatchV1(std::unique_ptr<BatchImpl> value) : impl(std::move(value)) {}
    std::unique_ptr<BatchImpl> impl;
};

extern "C" {

std::uint32_t ATOMIC_V3_PROVIDER_CDECL atomic_v3_provider_abi_version() {
    return AtomicV3ProviderAbiVersion;
}

const char* ATOMIC_V3_PROVIDER_CDECL atomic_v3_provider_creation_error() {
    return CreationError.data();
}

AtomicV3ProviderStatus ATOMIC_V3_PROVIDER_CDECL
atomic_v3_provider_create(const AtomicV3ProviderConfigV1* config,
                          AtomicV3ProviderStreamV1**       output) {
    CreationError[0] = '\0';
    if (output != nullptr)
        *output = nullptr;
    if (config == nullptr || output == nullptr)
    {
        set_creation_error("Atomic V3 provider config/output must not be null");
        return ATOMIC_V3_PROVIDER_ERROR;
    }
    if (config->abiVersion != AtomicV3ProviderAbiVersion
        || config->structSize != sizeof(AtomicV3ProviderConfigV1)
        || config->batchSize == 0 || config->batchSize > MaximumBatchSize
        || config->nativeWorkers != 1
        || (config->cyclic != 0 && config->cyclic != 1)
        || std::any_of(std::begin(config->reserved), std::end(config->reserved),
                       [](std::uint8_t value) { return value != 0; }))
    {
        set_creation_error("Atomic V3 provider ABI/configuration is invalid");
        return ATOMIC_V3_PROVIDER_ERROR;
    }
    try
    {
        *output = new AtomicV3ProviderStreamV1(*config);
        return ATOMIC_V3_PROVIDER_OK;
    }
    catch (const std::exception& exception)
    {
        set_creation_error(exception.what());
        return ATOMIC_V3_PROVIDER_ERROR;
    }
    catch (...)
    {
        set_creation_error("unknown Atomic V3 provider creation error");
        return ATOMIC_V3_PROVIDER_ERROR;
    }
}

void ATOMIC_V3_PROVIDER_CDECL atomic_v3_provider_destroy(AtomicV3ProviderStreamV1* stream) {
    delete stream;
}

AtomicV3ProviderStatus ATOMIC_V3_PROVIDER_CDECL
atomic_v3_provider_fetch(AtomicV3ProviderStreamV1* stream, AtomicV3ProviderBatchV1** output) {
    if (output != nullptr)
        *output = nullptr;
    if (stream == nullptr || output == nullptr)
        return ATOMIC_V3_PROVIDER_ERROR;
    try
    {
        std::unique_ptr<BatchImpl> batch = stream->impl.next();
        if (!batch)
            return ATOMIC_V3_PROVIDER_EOF;
        *output = new AtomicV3ProviderBatchV1(std::move(batch));
        return ATOMIC_V3_PROVIDER_OK;
    }
    catch (const std::exception& exception)
    {
        stream->impl.set_error(exception.what());
        return ATOMIC_V3_PROVIDER_ERROR;
    }
    catch (...)
    {
        stream->impl.set_error("unknown Atomic V3 provider fetch error");
        return ATOMIC_V3_PROVIDER_ERROR;
    }
}

const char* ATOMIC_V3_PROVIDER_CDECL
atomic_v3_provider_error(const AtomicV3ProviderStreamV1* stream) {
    return stream == nullptr ? "null Atomic V3 provider stream" : stream->impl.last_error().c_str();
}

AtomicV3ProviderStatus ATOMIC_V3_PROVIDER_CDECL
atomic_v3_provider_batch_view(const AtomicV3ProviderBatchV1* batch,
                              AtomicV3BatchViewV1*            output) {
    if (batch == nullptr || batch->impl == nullptr || output == nullptr)
        return ATOMIC_V3_PROVIDER_ERROR;
    *output = batch->impl->view();
    return ATOMIC_V3_PROVIDER_OK;
}

void ATOMIC_V3_PROVIDER_CDECL atomic_v3_provider_destroy_batch(AtomicV3ProviderBatchV1* batch) {
    delete batch;
}

AtomicV3ProviderStatus ATOMIC_V3_PROVIDER_CDECL
atomic_v3_provider_commit(AtomicV3ProviderStreamV1* stream) {
    if (stream == nullptr)
        return ATOMIC_V3_PROVIDER_ERROR;
    try
    {
        stream->impl.commit();
        return ATOMIC_V3_PROVIDER_OK;
    }
    catch (const std::exception& exception)
    {
        stream->impl.set_error(exception.what());
        return ATOMIC_V3_PROVIDER_ERROR;
    }
    catch (...)
    {
        stream->impl.set_error("unknown Atomic V3 provider commit error");
        return ATOMIC_V3_PROVIDER_ERROR;
    }
}

AtomicV3ProviderStatus ATOMIC_V3_PROVIDER_CDECL
atomic_v3_provider_committed_cursor(const AtomicV3ProviderStreamV1* stream,
                                    AtomicV3ProviderCursorV1*       output) {
    if (stream == nullptr || output == nullptr)
        return ATOMIC_V3_PROVIDER_ERROR;
    *output = stream->impl.committed_cursor();
    return ATOMIC_V3_PROVIDER_OK;
}

}  // extern "C"

static_assert(sizeof(AtomicV3ProviderCursorV1) == 88);
