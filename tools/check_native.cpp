// Optional compatibility oracle. Uses the unmodified native Prover and CPU verifier.
#include <mmx/pos/Prover.h>
#include <mmx/pos/verify.h>
#include <mmx/ProofOfSpaceOG.hxx>
#include <mmx/ProofOfSpaceNFT.hxx>
#include <vnx/vnx.h>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if(argc < 2) throw std::runtime_error("usage: check_native plot [sample-count]");
        mmx::secp256k1_init();
        mmx::pos::Prover prover(argv[1]);
        auto h = prover.get_header();
        if(h->contract) {
            auto identity = mmx::ProofOfSpaceNFT::create();
            identity->ksize = h->ksize; identity->seed = h->seed;
            identity->farmer_key = h->farmer_key; identity->contract = *h->contract;
            identity->plot_id = h->plot_id; identity->validate();
        } else {
            auto identity = mmx::ProofOfSpaceOG::create();
            identity->ksize = h->ksize; identity->seed = h->seed;
            identity->farmer_key = h->farmer_key; identity->plot_id = h->plot_id;
            identity->validate();
        }
        const uint64_t samples = argc > 2 ? std::stoull(argv[2]) : 16;
        if(!h->num_entries_y || !samples) throw std::runtime_error("empty plot or sample count");
        const auto count = std::min(samples, h->num_entries_y);
        for(uint64_t sample = 0; sample < count; ++sample) {
            const auto index = sample * h->num_entries_y / count;
            const auto proof = prover.get_full_proof(index);
            if(!proof.valid || proof.proof.size() != 256) throw std::runtime_error("invalid full proof");
            const auto entries = mmx::pos::compute(proof.proof, nullptr, h->plot_id, h->ksize, 0);
            if(entries.size() != 1) throw std::runtime_error("not a unique proof");
            // Search for exactly this Y. Also exercises the native park search,
            // final-entry indexing and HDD metadata decoding.
            mmx::hash_t challenge;
            const uint32_t y = entries[0].first;
            for(size_t j = 0; j < 4; ++j) challenge[j] = y >> (8 * j);
            mmx::pos::verify(proof.proof, challenge, h->plot_id, 0, 0, h->ksize, true);
            bool found = false;
            for(const auto& quality : prover.get_qualities(challenge, 0)) {
                if(quality.index == index && quality.valid && quality.meta == proof.meta) found = true;
            }
            if(!found) throw std::runtime_error("native quality search/metadata mismatch");
        }
        std::cout << "Native reader, qualities and CPU verifier passed " << count << " samples / " << h->num_entries_y << " entries\n";
        return 0;
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
