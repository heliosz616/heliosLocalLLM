#include "checkpoint-policy.h"
#include <cassert>
#include <iostream>
using namespace recurrent_checkpoint;
int main() {
    std::vector<int64_t> all;
    for (int64_t p=32768; p<=262144; p+=32768) all.push_back(p);
    auto retained=retained_positions(all);
    assert(retained.size()==6 && *retained.begin()==98304);
    assert(restore_position(retained,196606,196606,196608)==163840);
    assert(restore_position(retained,196608,196608,196608)==163840);
    assert(restore_position(retained,196608,196609,196608)==196608);
    assert(restore_position(retained,100,100,196608)==0);
    assert(restore_position(retained,196608,196609,0)==0);
    assert(capture_due(0,32768) && !capture_due(0,32767));
    assert(capture_due(32768,65600) && !capture_due(32768,40000));
    assert(boundaries(32767,65537)==std::vector<int64_t>({32768,65536}));
    bool rejected=false; try { retained_positions({0}); } catch(const std::invalid_argument&) {rejected=true;}
    assert(rejected);
    // Rewind A at 32K, then rebuild its suffix: A's old 64K checkpoint must
    // never be paired with B's recomputed KV, even if the tokens match again.
    const std::vector<std::pair<std::string,int64_t>> old={{"A",32768},{"A",65536}};
    auto b=inherited_prefixes("A",{},32768,old);
    assert(lineage_allows("B",b,"A",32768));
    assert(!lineage_allows("B",b,"A",65536));
    auto c=inherited_prefixes("B",b,63000,{{"A",32768},{"A",65536},{"B",66000}});
    assert(c.size()==1 && c.at("A")==32768);
    assert(!lineage_allows("C",c,"B",66000));
    assert(!lineage_allows("C",c,"A",65536));
    assert(inherited_prefixes("C",c,0,old).empty());
    auto recent=retained_lru({{65536,"old64"},{98304,"old96"},{131072,"old128"},
        {163840,"old160"},{196608,"old192"},{32768,"reused32"},{65996,"rebuilt66"}});
    assert(recent.size()==6 && recent.count("reused32") && recent.count("rebuilt66") && !recent.count("old64"));
    assert(retained_lru({{32768,"A"},{32768,"B"}})==std::set<std::string>{"B"});
    // Pending commits must reserve retention slots before publication.
    auto pending=retained_lru({{1,"a"},{2,"b"},{3,"c"},{4,"d"},{5,"e"},{6,"f"},
                               {7,"pending:0"},{8,"pending:1"}});
    assert(pending.size()==6 && !pending.count("a") && !pending.count("b"));
    assert(pending.count("pending:0") && pending.count("pending:1"));

    std::cout<<"32K/six retention and checkpoint restore selection passed\n";
}
