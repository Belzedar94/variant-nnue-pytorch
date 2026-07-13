#ifndef _SFEN_STREAM_H_
#define _SFEN_STREAM_H_

#include "nnue_training_data_formats.h"
#include "atomic_bin_v2_training_data.h"

#include <optional>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <memory>

namespace training_data {

    using namespace bin;

    static bool ends_with(const std::string& lhs, const std::string& end)
    {
        if (end.size() > lhs.size()) return false;

        return std::equal(end.rbegin(), end.rend(), lhs.rbegin());
    }

    static bool has_extension(const std::string& filename, const std::string& extension)
    {
        return ends_with(filename, "." + extension);
    }

    static std::string filename_with_extension(const std::string& filename, const std::string& ext)
    {
        if (ends_with(filename, ext))
        {
            return filename;
        }
        else
        {
            return filename + "." + ext;
        }
    }

    static bool is_atomic_bin_v2_manifest(const std::string& filename)
    {
        return ends_with(filename, ".atbin.manifest.json");
    }

    struct BasicSfenInputStream
    {
        virtual std::optional<TrainingDataEntry> next() = 0;
        virtual void fill(std::vector<TrainingDataEntry>& vec, std::size_t n)
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                auto v = this->next();
                if (!v.has_value())
                {
                    break;
                }
                vec.emplace_back(*v);
            }
        }

        virtual bool eof() const = 0;
        virtual ~BasicSfenInputStream() {}
    };

    struct BinSfenInputStream : BasicSfenInputStream
    {
        static constexpr auto openmode = std::ios::in | std::ios::binary;
        static inline const std::string extension = "bin";

        BinSfenInputStream(std::string filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
            m_stream(filename, openmode),
            m_filename(filename),
            m_eof(!m_stream),
            m_cyclic(cyclic),
            m_skipPredicate(std::move(skipPredicate))
        {
        }

        std::optional<TrainingDataEntry> next() override
        {
            nodchip::PackedSfenValue e;
            bool reopenedFileOnce = false;
            for(;;)
            {
                if(m_stream.read(reinterpret_cast<char*>(&e), sizeof(nodchip::PackedSfenValue)))
                {
                    TrainingDataEntry entry;
                    try
                    {
                        entry = packedSfenValueToTrainingDataEntry(e);
                    }
                    catch (const std::exception& error)
                    {
                        throw std::invalid_argument(
                            "invalid Legacy72 record " + std::to_string(m_record_index)
                            + ": " + error.what()
                        );
                    }
                    ++m_record_index;
                    if (!m_skipPredicate || !m_skipPredicate(entry))
                        return entry;
                }
                else
                {
                    if (m_cyclic)
                    {
                        if (reopenedFileOnce)
                            return std::nullopt;

                        m_stream = std::fstream(m_filename, openmode);
                        reopenedFileOnce = true;
                        if (!m_stream)
                            return std::nullopt;

                        continue;
                    }

                    m_eof = true;
                    return std::nullopt;
                }
            }
        }

        bool eof() const override
        {
            return m_eof;
        }

        ~BinSfenInputStream() override {}

    private:
        std::fstream m_stream;
        std::string m_filename;
        bool m_eof;
        bool m_cyclic;
        std::function<bool(const TrainingDataEntry&)> m_skipPredicate;
        std::uint64_t m_record_index{0};
    };

    struct AtomicBinV2InputStream : BasicSfenInputStream
    {
        AtomicBinV2InputStream(
            const std::string& manifest_filename,
            bool cyclic,
            std::function<bool(const TrainingDataEntry&)> skipPredicate) :
            m_cyclic(cyclic),
            m_skipPredicate(std::move(skipPredicate))
        {
            Stockfish::Data::DataResult opened =
                Stockfish::Data::AtomicBinV2DatasetReader::open(
                    std::filesystem::u8path(manifest_filename),
                    m_reader);
            if (!opened)
                throw std::invalid_argument(
                    "cannot open Atomic BIN V2 manifest: " + opened.message);

            const auto& options = m_reader->manifest().options;
            std::clog
                << "Atomic BIN V2 manifest policy: use_nnue=pure eval_limit=" << options.evalLimit
                << " filter_captures=" << (options.filterCaptures ? "true" : "false")
                << " filter_promotions=" << (options.filterPromotions ? "true" : "false")
                << " filter_checks=" << (options.filterChecks ? "true" : "false")
                << '\n';
        }

        std::optional<TrainingDataEntry> next() override
        {
            bool rewound_once = false;
            for (;;)
            {
                Stockfish::Data::AtomicBinV2DecodedRecord decoded{};
                bool has_record = false;
                Stockfish::Data::DataResult read = m_reader->next(decoded, has_record);
                if (!read)
                    throw std::invalid_argument(
                        "invalid Atomic BIN V2 dataset: " + read.message);

                if (has_record)
                {
                    TrainingDataEntry entry =
                        atomic_bin_v2::to_training_data_entry(decoded);
                    if (!m_skipPredicate || !m_skipPredicate(entry))
                        return entry;
                    continue;
                }

                if (m_cyclic && !rewound_once)
                {
                    Stockfish::Data::DataResult rewound = m_reader->rewind();
                    if (!rewound)
                        throw std::invalid_argument(
                            "cannot rewind Atomic BIN V2 dataset: " + rewound.message);
                    rewound_once = true;
                    continue;
                }

                m_eof = true;
                return std::nullopt;
            }
        }

        bool eof() const override
        {
            return m_eof;
        }

    private:
        std::unique_ptr<Stockfish::Data::AtomicBinV2DatasetReader> m_reader;
        bool m_eof{false};
        bool m_cyclic;
        std::function<bool(const TrainingDataEntry&)> m_skipPredicate;
    };

    inline std::unique_ptr<BasicSfenInputStream> open_sfen_input_file(const std::string& filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate = nullptr)
    {
        if (has_extension(filename, BinSfenInputStream::extension))
            return std::make_unique<BinSfenInputStream>(filename, cyclic, std::move(skipPredicate));

        if (is_atomic_bin_v2_manifest(filename))
            return std::make_unique<AtomicBinV2InputStream>(filename, cyclic, std::move(skipPredicate));

        if (has_extension(filename, "atbin"))
            throw std::invalid_argument(
                "Atomic BIN V2 raw shards are not dataset entrypoints; use the .atbin.manifest.json sidecar");

        return nullptr;
    }

    inline std::unique_ptr<BasicSfenInputStream> open_sfen_input_file_parallel(int concurrency, const std::string& filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate = nullptr)
    {
        // TODO (low priority): optimize and parallelize .bin reading.
        (void) concurrency;
        return open_sfen_input_file(filename, cyclic, std::move(skipPredicate));
    }
}

#endif
