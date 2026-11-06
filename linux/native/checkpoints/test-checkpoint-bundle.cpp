#include "checkpoint-bundle.h"
#include <cassert>
#include <iostream>
int main(int argc,char**argv){
 assert(argc==2); recurrent_checkpoint_bundle b;b.n_tokens=32768;b.pos_min=32767;b.pos_max=32767;
 b.tokens={1,0,0,0};b.target={2,3,4};b.draft={5,6};b.speculative={7};b.write(argv[1]);
 auto c=recurrent_checkpoint_bundle::read(argv[1],b.size());
 assert(c.n_tokens==b.n_tokens && c.tokens==b.tokens && c.target==b.target && c.draft==b.draft && c.speculative==b.speculative);
 bool failed=false;try{recurrent_checkpoint_bundle::read(argv[1],b.size()-1);}catch(const std::runtime_error&){failed=true;}assert(failed);
 std::cout<<"checkpoint bundle round-trip and invalid extent rejected\n";
}
