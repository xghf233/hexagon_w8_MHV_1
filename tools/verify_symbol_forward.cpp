// Exact forward-prefix verification, no floating point or arbitrary evaluation.
// Python proves a <2^120 absolute intermediate bound before invoking this code.
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

using Value = __int128_t;
struct Edge { int letter, parent; int64_t coefficient; };
using State = std::vector<std::pair<int, Value>>;
void need(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
uint64_t little(const unsigned char* p, int length) {
    uint64_t value=0;
    for(int i=0;i<length;++i) value |= uint64_t(p[i]) << (8*i);
    return value;
}
struct Layer {
    std::vector<Value> values;
    std::vector<unsigned char> marked;
    std::array<std::vector<int>,9> touched;
    explicit Layer(int nodes):values(nodes*9,0),marked(nodes*9,0){}
};
class Verify {
public:
    int weight=0,nodes=0;
    std::vector<std::vector<Edge>> inverse;
    std::array<std::vector<int64_t>,9> final;
    std::vector<Layer> layers;
    std::unordered_set<uint64_t> prefixes;
    std::ifstream reference;
    bool compare=false,filtered=false;
    uint64_t expected=0,rows=0,visited=0,zeros=0;
    double seconds_limit=300;
    std::chrono::steady_clock::time_point started=std::chrono::steady_clock::now();
    double elapsed() const {return std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();}
    void load(const std::string& path) {
        std::ifstream input(path);
        std::string magic; uint64_t edges=0,terms=0;
        input>>magic>>weight>>nodes>>edges>>terms;
        need(magic=="HX5V1" && weight>=2 && weight<=10 && nodes>0 && nodes<=3000,"Invalid graph header");
        need(edges<=3000000 && terms<=30000,"Graph size limit");
        std::vector<int> weights(nodes);
        for(int& w:weights) input>>w;
        need(weights[0]==0,"Invalid scalar node");
        inverse.resize(nodes);
        for(uint64_t i=0;i<edges;++i) {
            int child,letter,parent; int64_t c;
            input>>child>>letter>>parent>>c;
            need(input.good() && child>=0 && child<nodes && parent>0 && parent<nodes && letter>=0 && letter<9,"Invalid edge");
            need(weights[parent]==weights[child]+1 && weights[parent]<weight,"Nonlowering edge");
            inverse[child].push_back({letter,parent,c});
        }
        for(auto& row:final) row.assign(nodes,0);
        for(uint64_t i=0;i<terms;++i) {
            int letter,child; int64_t c;
            input>>letter>>child>>c;
            need(input.good() && letter>=0 && letter<9 && child>0 && child<nodes && weights[child]==weight-1,"Invalid projection");
            need(final[letter][child]==0,"Duplicate projection");
            final[letter][child]=c;
        }
        std::string extra; need(!(input>>extra),"Trailing graph input");
        for(int i=0;i<weight;++i) layers.emplace_back(nodes);
    }
    void open_reference(const std::string& path) {
        if(path=="-") return;
        compare=true;
        reference.open(path,std::ios::binary);
        char magic[8]; unsigned char count[8];
        reference.read(magic,8); reference.read(reinterpret_cast<char*>(count),8);
        need(reference.good() && std::string(magic,8)=="HX5B001\n","Invalid binary reference");
        expected=little(count,8);
    }
    void emit(uint64_t word,Value value) {
        if(!value) {++zeros;return;}
        need(word<=std::numeric_limits<uint32_t>::max(),"Word index overflow");
        need(value>=std::numeric_limits<int64_t>::min() && value<=std::numeric_limits<int64_t>::max(),"Coefficient output overflow");
        if(compare) {
            unsigned char record[12];
            reference.read(reinterpret_cast<char*>(record),12);
            need(reference.good(),"Missing reference row");
            const uint32_t ref_word=uint32_t(little(record,4));
            const uint64_t bits=little(record+4,8);
            int64_t ref_value; std::memcpy(&ref_value,&bits,8);
            need(ref_word==word && Value(ref_value)==value,"Word/coefficient mismatch");
        }
        ++rows;
    }
    void walk(uint64_t prefix,int depth,const State& state) {
        if(filtered && depth==3 && !prefixes.contains(prefix)) return;
        ++visited;
        if(visited%65536==0) need(elapsed()<seconds_limit,"Native verification time limit");
        if(depth==weight-1) {
            for(int letter=0;letter<9;++letter) {
                Value value=0;
                for(const auto& [node,c]:state) value += c * Value(final[letter][node]);
                emit(prefix*9+letter,value);
            }
            return;
        }
        Layer& layer=layers[depth];
        for(const auto& [child,scalar]:state) for(const Edge& e:inverse[child]) {
            const int index=e.letter*nodes+e.parent;
            if(!layer.marked[index]) {layer.marked[index]=1;layer.touched[e.letter].push_back(e.parent);}
            layer.values[index] += scalar * Value(e.coefficient);
        }
        for(int letter=0;letter<9;++letter) {
            State next;
            next.reserve(layer.touched[letter].size());
            for(int parent:layer.touched[letter]) {
                const int index=letter*nodes+parent;
                if(layer.values[index]) next.emplace_back(parent,layer.values[index]);
                layer.values[index]=0;layer.marked[index]=0;
            }
            layer.touched[letter].clear();
            if(!next.empty()) walk(prefix*9+letter,depth+1,next);
        }
    }
    void run() {
        walk(0,0,State{{0,1}});
        if(compare) {
            need(rows==expected,"Reference row count mismatch");
            char extra; need(!reference.get(extra),"Extra reference bytes");
        }
        std::cout<<"{\"rows\":"<<rows<<",\"reachable_prefixes\":"<<visited
                 <<",\"zero_final_projections\":"<<zeros<<",\"seconds\":"<<elapsed()
                 <<",\"complete_support_match\":"<<(compare?"true":"false")<<"}\n";
    }
};
int main(int argc,char** argv) {
    try {
        need(argc==5,"Usage: verifier graph reference-or-dash prefix-list-or-all seconds");
        Verify verifier;
        verifier.load(argv[1]);verifier.open_reference(argv[2]);
        verifier.seconds_limit=std::stod(argv[4]);
        need(verifier.seconds_limit>0 && verifier.seconds_limit<=600,"Invalid runtime limit");
        if(std::string(argv[3])!="all") {
            verifier.filtered=true;std::stringstream in(argv[3]);std::string part;
            while(std::getline(in,part,',')) {auto code=std::stoull(part);need(code<729,"Invalid pilot prefix");verifier.prefixes.insert(code);}
        }
        need(!(verifier.compare && verifier.filtered),"Cannot claim full verification with prefix filter");
        verifier.run();return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<"\n";return 1;}
}
