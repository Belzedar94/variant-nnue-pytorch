#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/stat.h>
#endif

#include "dataset_file_identity.h"

namespace training_data::platform
{
    std::optional<DatasetFileIdentity> dataset_file_identity(
        const std::filesystem::path& path) noexcept
    {
#if defined(_WIN32)
        const HANDLE handle = CreateFileW(
            path.c_str(),
            0,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            nullptr,
            OPEN_EXISTING,
            FILE_FLAG_BACKUP_SEMANTICS,
            nullptr);
        if (handle == INVALID_HANDLE_VALUE)
            return std::nullopt;
        BY_HANDLE_FILE_INFORMATION information{};
        const bool inspected = bool(GetFileInformationByHandle(handle, &information));
        CloseHandle(handle);
        if (!inspected)
            return std::nullopt;
        return DatasetFileIdentity{
            information.dwVolumeSerialNumber,
            (std::uint64_t(information.nFileIndexHigh) << 32) | information.nFileIndexLow};
#else
        struct stat information{};
        if (::stat(path.c_str(), &information) != 0)
            return std::nullopt;
        return DatasetFileIdentity{
            static_cast<std::uint64_t>(information.st_dev),
            static_cast<std::uint64_t>(information.st_ino)};
#endif
    }
}
