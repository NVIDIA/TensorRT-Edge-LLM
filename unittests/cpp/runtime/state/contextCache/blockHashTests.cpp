/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "runtime/state/contextCache/blockHash.h"
#include "runtime/state/contextCache/blockIndex.h"
#include "runtime/state/contextCache/cacheRecord.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

using namespace trt_edgellm::rt;

namespace
{

constexpr BlockHash kPARENT_HASH{0x1020304050607080ULL, 0x90A0B0C0D0E0F001ULL};

BlockKeyExtras makeExtras()
{
    BlockKeyExtras extras;
    extras.media.push_back(MediaSpanKey{Hash128{0x1112131415161718ULL, 0x2122232425262728ULL},
        Hash128{0x3132333435363738ULL, 0x4142434445464748ULL}, 3, 5, 2});
    extras.media.push_back(MediaSpanKey{Hash128{0xD1D2D3D4D5D6D7D8ULL, 0xE1E2E3E4E5E6E7E8ULL},
        Hash128{0xF1F2F3F4F5F6F7F8ULL, 0x0102030405060708ULL}, -4, 9, 3});
    extras.adapter = AdapterKey{Hash128{0x5152535455565758ULL, 0x6162636465666768ULL}, 7};
    extras.positionDigest = Hash128{0x7172737475767778ULL, 0x8182838485868788ULL};
    extras.customEmbeddingDigest = Hash128{0x9192939495969798ULL, 0xA1A2A3A4A5A6A7A8ULL};
    extras.isolationDigest = Hash128{0xB1B2B3B4B5B6B7B8ULL, 0xC1C2C3C4C5C6C7C8ULL};
    return extras;
}

SpecPagedStateRecord makePagedSpecState(std::vector<PageId> pagePath)
{
    return SpecPagedStateRecord{std::move(pagePath)};
}

} // namespace

TEST(ContextCacheBlockHashTests, EqualInputsProduceEqualHash)
{
    std::array<int32_t, 4> const tokens{17, -3, 42, 0};
    BlockKeyExtras const extras = makeExtras();

    BlockHash const first = hashBlock(kPARENT_HASH, tokens.data(), tokens.size(), extras);
    BlockHash const second = hashBlock(kPARENT_HASH, tokens.data(), tokens.size(), extras);
    EXPECT_EQ(first, second);
    // This value freezes cache identity; changing it is a cache-identity compatibility change.
    BlockHash const expected{0xBA9FE04E33CEF5BDULL, 0xD4A509A375759518ULL};
    EXPECT_EQ(first, expected);

    std::array<int32_t, 4> const changedTokens{17, -3, 43, 0};
    EXPECT_NE(first, hashBlock(kPARENT_HASH, changedTokens.data(), changedTokens.size(), extras));

    // These fixed non-colliding probes lock both halves of the folding implementation.
    Hash128 const foldInput{0x0102030405060708ULL, 0x1112131415161718ULL};
    Hash128 const changedHi{0x2122232425262728ULL, 0x1112131415161718ULL};
    Hash128 const changedLo{0x0102030405060708ULL, 0x3132333435363738ULL};
    size_t const folded = std::hash<Hash128>{}(foldInput);
    EXPECT_NE(folded, std::hash<Hash128>{}(changedHi));
    EXPECT_NE(folded, std::hash<Hash128>{}(changedLo));
}

TEST(ContextCacheBlockHashTests, OpaqueIdentityHashBindsExactBytes)
{
    std::string_view const first{"adapter-a"};
    std::string_view const second{"adapter-b"};

    EXPECT_EQ(hashOpaqueIdentity(first), hashOpaqueIdentity(first));
    EXPECT_NE(hashOpaqueIdentity(first), hashOpaqueIdentity(second));
    EXPECT_NE(hashOpaqueIdentity(first), hashOpaqueIdentity(std::string_view{"adapter-a\0", 10}));
    EXPECT_NE(hashOpaqueIdentity({}), Hash128{});
}

TEST(ContextCacheBlockHashTests, ParentHashChangesEveryDescendant)
{
    std::array<int32_t, 2> const firstTokens{11, 12};
    std::array<int32_t, 2> const secondTokens{21, 22};
    std::array<int32_t, 2> const thirdTokens{31, 32};
    BlockHash const otherRoot{0xDEADBEEF01234567ULL, 0x89ABCDEF76543210ULL};

    BlockHash const first = hashBlock(kCHAIN_ROOT, firstTokens.data(), firstTokens.size());
    BlockHash const second = hashBlock(first, secondTokens.data(), secondTokens.size());
    BlockHash const third = hashBlock(second, thirdTokens.data(), thirdTokens.size());

    BlockHash const otherFirst = hashBlock(otherRoot, firstTokens.data(), firstTokens.size());
    BlockHash const otherSecond = hashBlock(otherFirst, secondTokens.data(), secondTokens.size());
    BlockHash const otherThird = hashBlock(otherSecond, thirdTokens.data(), thirdTokens.size());

    EXPECT_EQ(first, hashBlock(kCHAIN_ROOT, firstTokens.data(), firstTokens.size()));
    EXPECT_NE(first, otherFirst);
    EXPECT_NE(second, otherSecond);
    EXPECT_NE(third, otherThird);
}

TEST(ContextCacheBlockHashTests, AdapterGenerationChangesHash)
{
    std::array<int32_t, 3> const tokens{4, 8, 15};
    BlockKeyExtras firstExtras;
    firstExtras.adapter = AdapterKey{Hash128{0x1111222233334444ULL, 0x5555666677778888ULL}, 19};
    BlockKeyExtras nextGenerationExtras = firstExtras;
    nextGenerationExtras.adapter->generation = 20;
    BlockKeyExtras otherAdapterExtras = firstExtras;
    otherAdapterExtras.adapter->id.lo = 0x9999AAAABBBBCCCCULL;

    BlockHash const first = hashBlock(kPARENT_HASH, tokens.data(), tokens.size(), firstExtras);
    EXPECT_EQ(first, hashBlock(kPARENT_HASH, tokens.data(), tokens.size(), firstExtras));
    EXPECT_NE(first, hashBlock(kPARENT_HASH, tokens.data(), tokens.size(), nextGenerationExtras));
    EXPECT_NE(first, hashBlock(kPARENT_HASH, tokens.data(), tokens.size(), otherAdapterExtras));

    BlockKeyExtras zeroAdapterExtras;
    zeroAdapterExtras.adapter = AdapterKey{};
    EXPECT_NE(hashBlock(kPARENT_HASH, tokens.data(), tokens.size()),
        hashBlock(kPARENT_HASH, tokens.data(), tokens.size(), zeroAdapterExtras));
}

TEST(ContextCacheBlockHashTests, MediaPlacementChangesHash)
{
    std::array<int32_t, 2> const tokens{23, 29};
    MediaSpanKey const firstSpan{Hash128{0x1011121314151617ULL, 0x2021222324252627ULL},
        Hash128{0x3031323334353637ULL, 0x4041424344454647ULL}, 2, 3, 1};
    MediaSpanKey const secondSpan{Hash128{0x5051525354555657ULL, 0x6061626364656667ULL},
        Hash128{0x7071727374757677ULL, 0x8081828384858687ULL}, 5, 7, 2};
    BlockKeyExtras extras;
    extras.media = {firstSpan, secondSpan};

    BlockHash const baseline = hashBlock(kPARENT_HASH, tokens.data(), tokens.size(), extras);
    EXPECT_EQ(baseline, hashBlock(kPARENT_HASH, tokens.data(), tokens.size(), extras));

    BlockKeyExtras changed = extras;
    changed.media.front().contentDigest.hi = 0x9091929394959697ULL;
    EXPECT_NE(baseline, hashBlock(kPARENT_HASH, tokens.data(), tokens.size(), changed));
    changed = extras;
    changed.media.front().placementDigest.lo = 0xA0A1A2A3A4A5A6A7ULL;
    EXPECT_NE(baseline, hashBlock(kPARENT_HASH, tokens.data(), tokens.size(), changed));
    changed = extras;
    changed.media.front().offsetWithinBlock = 4;
    EXPECT_NE(baseline, hashBlock(kPARENT_HASH, tokens.data(), tokens.size(), changed));
    changed = extras;
    changed.media.front().itemOrder = 6;
    EXPECT_NE(baseline, hashBlock(kPARENT_HASH, tokens.data(), tokens.size(), changed));
    changed = extras;
    changed.media.front().modality = 3;
    EXPECT_NE(baseline, hashBlock(kPARENT_HASH, tokens.data(), tokens.size(), changed));
    changed = extras;
    changed.media = {secondSpan, firstSpan};
    EXPECT_NE(baseline, hashBlock(kPARENT_HASH, tokens.data(), tokens.size(), changed));
}

TEST(ContextCacheBlockHashTests, PositionEmbeddingAndIsolationChangeHash)
{
    std::array<int32_t, 2> const tokens{37, 41};
    Hash128 const digest{0x0102030405060708ULL, 0x1112131415161718ULL};
    BlockKeyExtras positionExtras;
    positionExtras.positionDigest = digest;
    BlockKeyExtras embeddingExtras;
    embeddingExtras.customEmbeddingDigest = digest;
    BlockKeyExtras isolationExtras;
    isolationExtras.isolationDigest = digest;

    BlockHash const position = hashBlock(kPARENT_HASH, tokens.data(), tokens.size(), positionExtras);
    BlockHash const embedding = hashBlock(kPARENT_HASH, tokens.data(), tokens.size(), embeddingExtras);
    BlockHash const isolation = hashBlock(kPARENT_HASH, tokens.data(), tokens.size(), isolationExtras);

    EXPECT_EQ(position, hashBlock(kPARENT_HASH, tokens.data(), tokens.size(), positionExtras));
    EXPECT_NE(position, embedding);
    EXPECT_NE(position, isolation);
    EXPECT_NE(embedding, isolation);
    EXPECT_NE(position, hashBlock(kPARENT_HASH, tokens.data(), tokens.size()));

    BlockKeyExtras changedPositionExtras = positionExtras;
    changedPositionExtras.positionDigest->lo = 0x2122232425262728ULL;
    EXPECT_NE(position, hashBlock(kPARENT_HASH, tokens.data(), tokens.size(), changedPositionExtras));
    BlockKeyExtras changedEmbeddingExtras = embeddingExtras;
    changedEmbeddingExtras.customEmbeddingDigest->hi = 0x3132333435363738ULL;
    EXPECT_NE(embedding, hashBlock(kPARENT_HASH, tokens.data(), tokens.size(), changedEmbeddingExtras));
    BlockKeyExtras changedIsolationExtras = isolationExtras;
    changedIsolationExtras.isolationDigest->lo = 0x4142434445464748ULL;
    EXPECT_NE(isolation, hashBlock(kPARENT_HASH, tokens.data(), tokens.size(), changedIsolationExtras));

    BlockKeyExtras zeroPositionExtras;
    zeroPositionExtras.positionDigest = Hash128{};
    EXPECT_NE(hashBlock(kPARENT_HASH, tokens.data(), tokens.size()),
        hashBlock(kPARENT_HASH, tokens.data(), tokens.size(), zeroPositionExtras));
}

TEST(ContextCacheBlockHashTests, FullBlockChainIgnoresPartialTail)
{
    constexpr int32_t kPAGE_SIZE = 4;
    std::array<int32_t, 10> const tokens{2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
    std::array<int32_t, 10> changedTail = tokens;
    changedTail[8] = 31;
    changedTail[9] = 37;

    std::vector<BlockHash> const hashes = hashFullBlocks(tokens.data(), tokens.size(), kPAGE_SIZE);
    std::vector<BlockHash> const changedTailHashes = hashFullBlocks(changedTail.data(), changedTail.size(), kPAGE_SIZE);
    ASSERT_EQ(hashes.size(), 2U);
    EXPECT_EQ(hashes, changedTailHashes);

    BlockHash const first = hashBlock(kCHAIN_ROOT, tokens.data(), kPAGE_SIZE);
    BlockHash const second = hashBlock(first, tokens.data() + kPAGE_SIZE, kPAGE_SIZE);
    EXPECT_EQ(hashes[0], first);
    EXPECT_EQ(hashes[1], second);

    std::array<int32_t, 10> changedFullBlock = tokens;
    changedFullBlock[6] = 43;
    std::vector<BlockHash> const changedFullBlockHashes
        = hashFullBlocks(changedFullBlock.data(), changedFullBlock.size(), kPAGE_SIZE);
    ASSERT_EQ(changedFullBlockHashes.size(), hashes.size());
    EXPECT_EQ(changedFullBlockHashes[0], hashes[0]);
    EXPECT_NE(changedFullBlockHashes[1], hashes[1]);

    std::vector<BlockKeyExtras> extras(2);
    extras[1].isolationDigest = Hash128{0xABCDEF0123456789ULL, 0x9876543210FEDCBAULL};
    std::vector<BlockHash> const hashesWithExtras = hashFullBlocks(tokens.data(), tokens.size(), kPAGE_SIZE, extras);
    EXPECT_EQ(hashesWithExtras[0], hashes[0]);
    EXPECT_NE(hashesWithExtras[1], hashes[1]);

    EXPECT_TRUE(hashFullBlocks(nullptr, 0, kPAGE_SIZE).empty());
    EXPECT_TRUE(hashFullBlocks(nullptr, 3, kPAGE_SIZE).empty());
    EXPECT_EQ(hashBlock(kPARENT_HASH, nullptr, 0), hashBlock(kPARENT_HASH, nullptr, 0));
    EXPECT_THROW((void) hashBlock(kPARENT_HASH, nullptr, 1), std::runtime_error);
    EXPECT_THROW((void) hashFullBlocks(tokens.data(), tokens.size(), 0), std::runtime_error);
    EXPECT_THROW((void) hashFullBlocks(tokens.data(), tokens.size(), -1), std::runtime_error);
    EXPECT_THROW((void) hashFullBlocks(nullptr, kPAGE_SIZE, kPAGE_SIZE), std::runtime_error);
    EXPECT_THROW((void) hashFullBlocks(tokens.data(), tokens.size(), kPAGE_SIZE, std::vector<BlockKeyExtras>(1)),
        std::runtime_error);
}

TEST(ContextCacheBlockHashTests, ExactPrefixDigestIncludesPartialTail)
{
    constexpr int32_t kPAGE_SIZE = 4;
    std::array<int32_t, 6> const tokens{2, 3, 5, 7, 11, 13};
    std::array<int32_t, 6> changedTail = tokens;
    changedTail.back() = 17;

    BlockHash const fullBoundary = hashExactPrefix(tokens.data(), kPAGE_SIZE, kPAGE_SIZE);
    BlockHash const expectedFullBoundary = hashBlock(kCHAIN_ROOT, tokens.data(), kPAGE_SIZE);
    EXPECT_EQ(fullBoundary, expectedFullBoundary);

    BlockHash const partial = hashExactPrefix(tokens.data(), tokens.size(), kPAGE_SIZE);
    EXPECT_EQ(partial, hashBlock(expectedFullBoundary, tokens.data() + kPAGE_SIZE, 2));
    EXPECT_NE(partial, hashExactPrefix(changedTail.data(), changedTail.size(), kPAGE_SIZE));
    EXPECT_EQ(hashExactPrefix(nullptr, 0, kPAGE_SIZE), kCHAIN_ROOT);

    std::vector<BlockKeyExtras> extras(2);
    extras[1].isolationDigest = Hash128{19, 23};
    EXPECT_NE(partial, hashExactPrefix(tokens.data(), tokens.size(), kPAGE_SIZE, extras));
    EXPECT_THROW((void) hashExactPrefix(tokens.data(), tokens.size(), 0), std::runtime_error);
    EXPECT_THROW((void) hashExactPrefix(nullptr, 1, kPAGE_SIZE), std::runtime_error);
    EXPECT_THROW((void) hashExactPrefix(tokens.data(), tokens.size(), kPAGE_SIZE, std::vector<BlockKeyExtras>(1)),
        std::runtime_error);
}

TEST(ContextCacheBlockIndexTests, FirstCommitterOwnsCanonicalPage)
{
    BaseBlockIndex index;
    BlockHash const hash{0x5000000000000005ULL, 0x6000000000000006ULL};

    BaseInsertResult const first = index.insert(hash, 7);
    BaseInsertResult const second = index.insert(hash, 8);

    EXPECT_TRUE(first.inserted);
    EXPECT_FALSE(second.inserted);
    EXPECT_EQ(first.canonicalPage, 7);
    EXPECT_EQ(second.canonicalPage, 7);
    EXPECT_EQ(index.lookup(hash), std::optional<PageId>{7});
    EXPECT_EQ(index.size(), 1U);
}

TEST(ContextCacheBlockIndexTests, LookupStopsAtFirstMissingBlock)
{
    BaseBlockIndex index;
    BlockHash const firstHash{0x3333333333333333ULL, 0x4444444444444444ULL};
    BlockHash const missingHash{0x5555555555555555ULL, 0x6666666666666666ULL};
    BlockHash const laterHash{0x7777777777777777ULL, 0x8888888888888888ULL};
    EXPECT_TRUE(index.insert(firstHash, 13).inserted);
    EXPECT_TRUE(index.insert(laterHash, 17).inserted);

    BaseLookupResult const result = index.lookupPrefix({firstHash, missingHash, laterHash});

    EXPECT_EQ(result.pageIds, std::vector<PageId>{13});
    EXPECT_EQ(result.matchedHashes, std::vector<BlockHash>{firstHash});
    EXPECT_FALSE(index.lookup(missingHash).has_value());
    EXPECT_EQ(index.lookup(laterHash), std::optional<PageId>{17});
}

TEST(ContextCacheBlockIndexTests, LookupHasNoReferenceOrRecencySideEffects)
{
    BaseBlockIndex index;
    BlockHash const firstHash{0x5656565656565656ULL, 0x7878787878787878ULL};
    BlockHash const secondHash{0x9090909090909090ULL, 0xABABABABABABABABULL};
    EXPECT_TRUE(index.insert(firstHash, 19).inserted);
    EXPECT_TRUE(index.insert(secondHash, 23).inserted);
    size_t const originalSize = index.size();

    BaseLookupResult const firstLookup = index.lookupPrefix({firstHash, secondHash});
    EXPECT_EQ(index.lookup(firstHash), std::optional<PageId>{19});
    EXPECT_FALSE(index.lookup(Hash128{0xCDCDCDCDCDCDCDCDULL, 0xEFEFEFEFEFEFEFEFULL}).has_value());
    BaseLookupResult const secondLookup = index.lookupPrefix({firstHash, secondHash});

    EXPECT_EQ(firstLookup.pageIds, std::vector<PageId>({19, 23}));
    EXPECT_EQ(firstLookup.matchedHashes, std::vector<BlockHash>({firstHash, secondHash}));
    EXPECT_EQ(secondLookup.pageIds, firstLookup.pageIds);
    EXPECT_EQ(secondLookup.matchedHashes, firstLookup.matchedHashes);
    EXPECT_EQ(index.size(), originalSize);

    BaseInsertResult const duplicate = index.insert(firstHash, 29);
    EXPECT_FALSE(duplicate.inserted);
    EXPECT_EQ(duplicate.canonicalPage, 19);
    EXPECT_EQ(index.lookup(secondHash), std::optional<PageId>{23});
}

TEST(ContextCacheBlockIndexTests, FirstCommitterWinsAndReverseEraseIsExact)
{
    BaseBlockIndex index;
    BlockHash const firstHash{0x1111222233334444ULL, 0x5555666677778888ULL};
    BlockHash const secondHash{0x9999AAAABBBBCCCCULL, 0xDDDDEEEEFFFF0000ULL};
    BlockHash const thirdHash{0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL};

    BaseInsertResult const first = index.insert(firstHash, 31);
    BaseInsertResult const second = index.insert(secondHash, 37);
    EXPECT_TRUE(first.inserted);
    EXPECT_TRUE(second.inserted);
    EXPECT_EQ(first.canonicalPage, 31);
    EXPECT_EQ(second.canonicalPage, 37);

    BaseInsertResult const duplicate = index.insert(firstHash, 41);
    EXPECT_FALSE(duplicate.inserted);
    EXPECT_EQ(duplicate.canonicalPage, 31);

    BaseInsertResult const duplicateProposalPage = index.insert(thirdHash, 41);
    EXPECT_TRUE(duplicateProposalPage.inserted);
    EXPECT_EQ(duplicateProposalPage.canonicalPage, 41);
    EXPECT_EQ(index.lookup(firstHash), std::optional<PageId>{31});
    EXPECT_EQ(index.lookup(thirdHash), std::optional<PageId>{41});
    index.erasePage(41);
    EXPECT_EQ(index.lookup(firstHash), std::optional<PageId>{31});
    EXPECT_FALSE(index.lookup(thirdHash).has_value());
    EXPECT_EQ(index.size(), 2U);

    BaseInsertResult const duplicateWithInvalidProposal = index.insert(firstHash, -1);
    EXPECT_FALSE(duplicateWithInvalidProposal.inserted);
    EXPECT_EQ(duplicateWithInvalidProposal.canonicalPage, 31);

    EXPECT_THROW((void) index.insert(thirdHash, 37), std::runtime_error);
    EXPECT_THROW((void) index.insert(thirdHash, -1), std::runtime_error);
    EXPECT_EQ(index.lookup(firstHash), std::optional<PageId>{31});
    EXPECT_EQ(index.lookup(secondHash), std::optional<PageId>{37});
    EXPECT_FALSE(index.lookup(thirdHash).has_value());

    index.erasePage(41);
    EXPECT_EQ(index.lookup(firstHash), std::optional<PageId>{31});
    EXPECT_EQ(index.lookup(secondHash), std::optional<PageId>{37});

    index.erasePage(31);
    EXPECT_FALSE(index.lookup(firstHash).has_value());
    EXPECT_EQ(index.lookup(secondHash), std::optional<PageId>{37});
    EXPECT_EQ(index.size(), 1U);

    BaseInsertResult const reusedPage = index.insert(thirdHash, 31);
    EXPECT_TRUE(reusedPage.inserted);
    EXPECT_EQ(reusedPage.canonicalPage, 31);
    index.erasePage(37);
    EXPECT_FALSE(index.lookup(secondHash).has_value());
    EXPECT_EQ(index.lookup(thirdHash), std::optional<PageId>{31});
    EXPECT_EQ(index.size(), 1U);
}

TEST(ContextCacheBlockIndexTests, SpecPagedLookupSelectsOneCoherentRecordPath)
{
    BlockHash const firstHash{0x0101010101010101ULL, 0x0202020202020202ULL};
    BlockHash const secondHash{0x0303030303030303ULL, 0x0404040404040404ULL};
    BlockHash const firstLeafHash{0x0505050505050505ULL, 0x0606060606060606ULL};
    BlockHash const secondLeafHash{0x0707070707070707ULL, 0x0808080808080808ULL};

    CacheRecord firstRecord;
    firstRecord.id = 11;
    firstRecord.key = CacheRecordKey{firstLeafHash, 3};
    firstRecord.logicalBlockHashes = {firstHash, secondHash, firstLeafHash};
    firstRecord.basePagePath = {1, 2, 3};
    firstRecord.specState = makePagedSpecState({4, 5, 6});

    CacheRecord secondRecord = firstRecord;
    secondRecord.id = 12;
    secondRecord.key = CacheRecordKey{secondLeafHash, 3};
    secondRecord.logicalBlockHashes.back() = secondLeafHash;
    secondRecord.basePagePath = {1, 2, 7};
    secondRecord.specState = makePagedSpecState({8, 9, 10});

    SpecPagedStateIndex index;
    CacheRecord incompleteRecord = firstRecord;
    incompleteRecord.id = 13;
    incompleteRecord.specState = makePagedSpecState({4, 5});
    EXPECT_THROW(index.insert(incompleteRecord), std::runtime_error);
    index.insert(firstRecord);
    index.insert(secondRecord);

    SpecPagedStateMatch const firstLeaf{firstRecord.id, 3};
    SpecPagedStateMatch const secondLeaf{secondRecord.id, 3};
    EXPECT_EQ(index.lookupLongest(firstRecord.logicalBlockHashes, 3), std::optional<SpecPagedStateMatch>{firstLeaf});
    EXPECT_EQ(index.lookupLongest(secondRecord.logicalBlockHashes, 3), std::optional<SpecPagedStateMatch>{secondLeaf});
    std::optional<SpecPagedStateMatch> const sharedPrefix = index.lookupLongest({firstHash, secondHash, Hash128{}}, 3);
    ASSERT_TRUE(sharedPrefix.has_value());
    EXPECT_EQ(sharedPrefix->pathBlockCount, 2);
    EXPECT_TRUE(sharedPrefix->record == firstRecord.id || sharedPrefix->record == secondRecord.id);
    EXPECT_TRUE(index.contains(secondHash, *sharedPrefix));

    index.erase(secondRecord);

    EXPECT_FALSE(index.contains(secondLeafHash, secondLeaf));
    EXPECT_EQ(index.lookupLongest({firstHash, secondHash}, 2),
        (std::optional<SpecPagedStateMatch>{SpecPagedStateMatch{firstRecord.id, 2}}));
    EXPECT_EQ(index.lookupLongest(firstRecord.logicalBlockHashes, 3), std::optional<SpecPagedStateMatch>{firstLeaf});

    index.erase(firstRecord);
    EXPECT_FALSE(index.lookupLongest(firstRecord.logicalBlockHashes, 3).has_value());
}

TEST(ContextCacheBlockIndexTests, SpecPagedIndexKeepsRepeatedHashesAsDistinctBoundaries)
{
    BlockHash const repeatedHash{0x0101010101010101ULL, 0x0202020202020202ULL};
    CacheRecord record;
    record.id = 11;
    record.key = CacheRecordKey{repeatedHash, 3};
    record.logicalBlockHashes = {repeatedHash, repeatedHash, repeatedHash};
    record.basePagePath = {1, 2, 3};
    record.specState = makePagedSpecState({4, 5, 6});

    SpecPagedStateIndex index;
    index.insert(record);

    EXPECT_TRUE(index.contains(repeatedHash, SpecPagedStateMatch{record.id, 1}));
    EXPECT_TRUE(index.contains(repeatedHash, SpecPagedStateMatch{record.id, 2}));
    EXPECT_TRUE(index.contains(repeatedHash, SpecPagedStateMatch{record.id, 3}));
    EXPECT_EQ(index.lookupLongest(record.logicalBlockHashes, 3),
        (std::optional<SpecPagedStateMatch>{SpecPagedStateMatch{record.id, 3}}));

    index.erase(record);
    EXPECT_FALSE(index.lookupLongest(record.logicalBlockHashes, 3).has_value());
}
