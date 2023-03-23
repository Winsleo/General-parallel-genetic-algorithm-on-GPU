#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <utility>
#define SHARED_PIECE_PRIORITY -10
#define BUDDY_PIECE_PRIORITY -1
//优先级队列的元素，候选碎片
struct ElemPriority{
    float priority;
    pair<int,int2> pieceId2Pos;
    int2 relative_piece;//(.x piece_id, .y orientation)
    
    ElemPriority():priority(0){}

    ElemPriority(float prior,pair<int,int2> id2pos,int2 rela_piece):priority(prior),pieceId2Pos(id2pos),relative_piece(rela_piece){}
    //重载运算符
    friend bool operator<(const ElemPriority& lhs,const ElemPriority& rhs){//priority默认是大顶堆
        return lhs.priority > rhs.priority;//变为小顶堆，优先级小的在前
    }
};
//二维位置转换为一维索引
int pos2idx(int2 pos){
    return pos.x*cols+pos.y;
}
class Crossover{
public:
    int* _parents[2];
    // Map piece ID to index in Individual's list
    vector<vector<int>> _piece_mapping;
    int _min_row;//kernel边界
    int _max_row;//kernel边界
    int _min_col;//kernel边界
    int _max_col;//kernel边界
    unordered_map<int,int2> _kernel;//kernel
    unordered_set<int> _taken_positions;//已经放置的位置
    priority_queue<ElemPriority> _candidate_pieces;//候选碎片优先队列

    Crossover(int* first_parent,int* second_parent){
        _parents[0] = first_parent;
        _parents[1] = second_parent;
        _piece_mapping.resize(2);
        _piece_mapping[0].resize(len);
        _piece_mapping[1].resize(len);
        for(int i=0;i<len;i++){
            _piece_mapping[0][_parents[0][i]]=i;
            _piece_mapping[1][_parents[1][i]]=i;
        }
        // Borders of growing kernel
        _min_row = 0;
        _max_row = 0;
        _min_col = 0;
        _max_col = 0;
    }
    //生成子代
    void child(int* son){
        for(auto id2pos:_kernel){
            int index= (id2pos.second.x - _min_row) * cols + (id2pos.second.y - _min_col);
            son[index] = id2pos.first;
        }
    }
    //运行主函数，执行交叉操作
    void run(){
        this->_initialize_kernel();//首先随机将一个碎片放入kernel

        while(_candidate_pieces.size() > 0){//有候选碎片
            auto candidata_piece = _candidate_pieces.top(); _candidate_pieces.pop();
            int piece_id = candidata_piece.pieceId2Pos.first;
            int2 position = candidata_piece.pieceId2Pos.second;
            int2 relative_piece = candidata_piece.relative_piece;
            if( _taken_positions.find( pos2idx(position) )!= _taken_positions.end() ){//如果该位置已经被放置，跳过本次循环执行下一次循环
                continue;
            }
            // If piece is already placed, find new piece candidate and put it back to
            // priority queue
            //如果该碎片已经被放置，寻找新的候选碎片并放入优先队列
            if (_kernel.find(piece_id) != _kernel.end() ){
                add_piece_candidate(relative_piece.x, relative_piece.y, position);
                continue;
            }
                
            //该碎片和该位置均空闲，放置该碎片到该位置
            _put_piece_to_kernel(piece_id, position);
        }
    }
    //随机生成初始kernel
    void _initialize_kernel(){
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed);
        int piece_id = _parents[0][generator()%len];
        _put_piece_to_kernel(piece_id, make_int2(0, 0));
    }
    //将碎片加入kernel，并从该碎片的邻居中选出候选碎片
    void _put_piece_to_kernel(int piece_id,int2 position){
        _kernel[piece_id] = position;
        _taken_positions.emplace( pos2idx(position) );//将位置设为已放置
        _update_candidate_pieces(piece_id, position);
    }
    //从position位置的邻居中选出候选碎片
    void _update_candidate_pieces(int piece_id, int2 position){
        auto available_boundaries = _available_boundaries(position);
        for(auto bound : available_boundaries){
            add_piece_candidate(piece_id, bound.first, bound.second);
        }         
    }
    //加入候选碎片
    void add_piece_candidate(int piece_id, int orientation,int2 position){
        //第一阶段
        // int shared_piece = _get_shared_piece(piece_id, orientation);//寻找父母共同的邻居
        // if (_is_valid_piece(shared_piece)){//如果该碎片是有效的
        //     _add_shared_piece_candidate( shared_piece, position, make_int2(piece_id, orientation) );
        //     return;
        // }
        //第二阶段
        int buddy_piece = _get_buddy_piece(piece_id, orientation);//寻找buddy碎片
        if ( _is_valid_piece(buddy_piece) ){
            _add_buddy_piece_candidate( buddy_piece, position, make_int2(piece_id, orientation) );
            return;
        }
        //第三阶段
        pair<int,float> best_match_piece = _get_best_match_piece(piece_id, orientation);//寻找该方向最匹配的碎片
        if (_is_valid_piece(best_match_piece.first)){
            _add_best_match_piece_candidate( best_match_piece.first, position, best_match_piece.second, make_int2(piece_id, orientation) );
            return;
        }
    }
    
    //寻找父母共同的邻居
    int _get_shared_piece(int piece_id, int orientation){
        //pieceid碎片在第一父母中orientation方向相邻的碎片id
        int first_parent_edge = edge(0,piece_id, orientation);
        //pieceid碎片在第二父母中orientation方向相邻的碎片id
        int second_parent_edge = edge(1,piece_id, orientation);
        if(first_parent_edge == second_parent_edge){
            return first_parent_edge;
        }
        return -1;
    }
    //寻找buddy碎片
    int _get_buddy_piece(int piece_id, int orientation){
        int first_buddy = best_match(piece_id, orientation).first;//id碎片在orientation方向最匹配的碎片
        int second_buddy = best_match(first_buddy,complementary_orientation(orientation)).first;

        if(second_buddy == piece_id){//如果互为最匹配对象
            if( first_buddy == edge(0,piece_id,orientation) || first_buddy == edge(1,piece_id,orientation) ){
                return first_buddy;
            }
        }
        return -1;
    }
    //寻找最优匹配碎片
    pair<int,float> _get_best_match_piece(int piece_id,int orientation){
        for(int i=0;i<len-1;i++){
            if(_is_valid_piece( best_match_table[orientation][piece_id][i].first ) ){
                return best_match_table[orientation][piece_id][i];
            }
        }
        return make_pair(-1,0);
    }
        
    void _add_shared_piece_candidate(int piece_id,int2 position,int2 relative_piece){//relative_piece(.x piece_id, .y orientation)
        ElemPriority piece_candidate(SHARED_PIECE_PRIORITY, make_pair(piece_id,position), relative_piece);
        _candidate_pieces.push(piece_candidate);
    }

    void _add_buddy_piece_candidate(int piece_id,int2 position,int2 relative_piece){//relative_piece(.x piece_id, .y orientation)
        ElemPriority piece_candidate(BUDDY_PIECE_PRIORITY, make_pair(piece_id,position), relative_piece);
        _candidate_pieces.push(piece_candidate);
    }

    void _add_best_match_piece_candidate(int piece_id,int2 position,float priority,int2 relative_piece){//relative_piece(.x piece_id, .y orientation)
        ElemPriority piece_candidate(priority, make_pair(piece_id,position), relative_piece);
        _candidate_pieces.push(piece_candidate);
    }

    //可选的kernel边界位置
    vector<pair<int,int2>> _available_boundaries(int2 position){
        vector<pair<int,int2>> boundaries;
        if( !_is_kernel_full()) {//如果还有位置没有放置碎片
            vector<pair<int,int2>> positions = {
                make_pair( 0, make_int2(position.x, position.y - 1) ),
                make_pair( 1, make_int2(position.x, position.y + 1) ),
                make_pair( 2, make_int2(position.x - 1, position.y) ),
                make_pair( 3, make_int2(position.x + 1, position.y) ),
            };

            for ( auto x : positions ) {
                if ( ( _taken_positions.find( pos2idx(x.second) )==_taken_positions.end() ) && _is_in_range(x.second) ){//如果position位置还没有被放置，且是合法的
                    _update_kernel_boundaries(x.second);//更新kernel边界
                    boundaries.push_back(x);//加入该位置与方向到返回列表
                }
            }  
        }
        return boundaries;
    }
    //kernel是否填满了所有位置
    bool _is_kernel_full(){
        return _kernel.size() == len;
    }
    //判断该位置是否合法
    bool _is_in_range(int2 position){
        return _is_row_in_range(position.x) && _is_column_in_range(position.y);
    }
    //判断该列是否合法
    bool _is_row_in_range(int row){
        int current_rows = abs(max(_max_row, row)-min(_min_row, row));
        return current_rows < rows;
    }
    //判断该行是否合法
    bool _is_column_in_range(int col){
        int current_columns = abs(max(_max_col, col)-min(_min_col, col));
        return current_columns < cols;
    }
    //更新kernel的边界
    void _update_kernel_boundaries(int2 position){
        _min_row = min(_min_row, position.x);
        _max_row = max(_max_row, position.x);
        _min_col = min(_min_col, position.y);
        _max_col = max(_max_col, position.y);
    }
    //是有效的碎片，还没有被放置
    bool _is_valid_piece(int piece_id){
        return (0 <= piece_id) && (piece_id<len) && (_kernel.find(piece_id)==_kernel.end());
    }
    // Left0 Right1 Top2 Down3
    int complementary_orientation(int orientation){//方向orientation的互补方向
        if(orientation==0) return 1;
        else if(orientation==1) return 0;
        else if (orientation==2) return 3;
        else if (orientation==3) return 2;
        else return -1;
    }
    // Left0 Right1 Top2 Down3
    int edge(int parentId,int piece_id,int orientation){
        int edge_index = _piece_mapping[parentId][piece_id];
        if( (orientation == 2) && (edge_index >= cols) ){
             return _parents[parentId][edge_index - cols];
        }
        if( (orientation == 1) && (edge_index % cols < cols - 1) ){
            return _parents[parentId][edge_index + 1];
        }   
        if( (orientation == 3) && (edge_index < (rows - 1) * cols) ){
            return _parents[parentId][edge_index + cols];
        } 
        if( (orientation == 0) && (edge_index % cols > 0) ){
            return _parents[parentId][edge_index - 1];
        }
        return -1;
    }
    pair<int,float> best_match(int piece_id,int orientation){
        return best_match_table[orientation][piece_id][0];//(.x 碎片id,.y 相异度)
    }
        
};
    
