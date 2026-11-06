#include "shared-cache-quota.h"
#include <cassert>
#include <iostream>
#include <sys/wait.h>
using path=std::filesystem::path;
static void write(const path & p,const std::string & v){std::ofstream f(p);f<<v;}
int main(int argc,char**argv){
 assert(argc==2);path root=argv[1];auto a=root/"a"/std::string(64,'a'), b=root/"b"/std::string(64,'b');
 std::filesystem::create_directories(a);std::filesystem::create_directories(b);
 shared_cache_quota quota({root/"a",root/"b"},root/"quota.lock",2000);
 auto r=quota.reserve(a,"1-1",1200);
 write(a/"1-1.part",std::string(900,'x'));
 assert(quota.inspect()<1500 && quota.inspect()>=1200);
 bool blocked=false;try{quota.reserve(b,"2-1",900);}catch(const std::runtime_error&){blocked=true;}assert(blocked);
 std::filesystem::rename(a/"1-1.part",a/"1-1.bin");
 write(a/"1-1.json",quota_json({{"version",1},{"identity",std::string(64,'a')},{"metadata",{{"session_id","session-A"}}}}).dump());
 quota.release(r);
 auto r2=quota.reserve(b,"2-2",1400);
 assert(!std::filesystem::exists(a/"1-1.bin"));assert(quota.inspect()<=2000);
 quota.release(r2);
 assert(quota.inspect()==0);
 auto abandoned=quota.reserve(a,"3-1",1000);
 write(a/"3-1.part",std::string(30,'x'));
 auto stale=shared_cache_quota::read_json(abandoned);stale["owner_start"]="stale-process";write(abandoned,stale.dump());
 quota.recover_abandoned();assert(!std::filesystem::exists(a/"3-1.part") && quota.inspect()==0);
 // Independent model writers cannot both admit a payload that exceeds the
 // shared capacity when combined, even before either writes its payload.
 pid_t children[2];
 for(int i=0;i<2;++i){
  children[i]=fork();assert(children[i]>=0);
  if(children[i]==0){
   try{quota.reserve(i ? b : a,"4-"+std::to_string(i),1500);_exit(0);}
   catch(const std::runtime_error&){_exit(2);}
  }
 }
 int admitted=0;
 for(auto pid:children){int status=0;assert(waitpid(pid,&status,0)==pid && WIFEXITED(status));
  assert(WEXITSTATUS(status)==0 || WEXITSTATUS(status)==2);admitted+=WEXITSTATUS(status)==0;}
 assert(admitted==1 && quota.inspect()<=2000);
 quota.recover_abandoned();assert(quota.inspect()==0);
 std::cout<<"shared-model budget, pending reservations and session eviction passed\n";
}
