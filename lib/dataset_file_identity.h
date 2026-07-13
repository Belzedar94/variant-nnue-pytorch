#ifndef DATASET_FILE_IDENTITY_H_
#define DATASET_FILE_IDENTITY_H_

#include <cstdint>
#include <filesystem>
#include <optional>

namespace training_data::platform
{
    struct DatasetFileIdentity
    {
        std::uint64_t first{};
        std::uint64_t second{};

        friend bool operator<(
            const DatasetFileIdentity& left,
            const DatasetFileIdentity& right) noexcept
        {
            return left.first < right.first
                || (left.first == right.first && left.second < right.second);
        }
    };

    std::optional<DatasetFileIdentity> dataset_file_identity(
        const std::filesystem::path& path) noexcept;
}

#endif
