#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <cassert>
#include <cstdio>
#include <functional>
using namespace std;
using namespace cv; //声明使用OpenCV的命名空间

//按列拼接
Mat mergeCols(Mat A, Mat B)
{
    assert(A.rows == B.rows&&A.type() == B.type());
    int totalCols = A.cols + B.cols;
 
    Mat mergedDescriptors(A.rows, totalCols, A.type());
    Mat submat = mergedDescriptors.colRange(0, A.cols);
    A.copyTo(submat);
    submat = mergedDescriptors.colRange(A.cols, totalCols);
    B.copyTo(submat);
    return mergedDescriptors;
}
//按行拼接
Mat mergeRows(Mat A, Mat B)
{
    assert(A.cols == B.cols&&A.type() == B.type());
    int totalRows = A.rows + B.rows;
 
    Mat mergedDescriptors(totalRows, A.cols, A.type());
    Mat submat = mergedDescriptors.rowRange(0, A.rows);
    A.copyTo(submat);
    submat = mergedDescriptors.rowRange(A.rows, totalRows);
    B.copyTo(submat);
    return mergedDescriptors;
 }
vector<Mat> flatten_image(Mat& image,int piece_size){
    /* 
    Converts image into list of square pieces.

    Input image is divided into square pieces of specified size and then
    flattened into list. Each list element is PIECE_SIZE x PIECE_SIZE x 3

    :params image:      Input image.
    :params piece_size: Size of single square piece. Each piece is PIECE_SIZE x PIECE_SIZE
    :params indexed:    If True list of Pieces with IDs will be returned, otherwise just plain list of ndarray pieces

    Usage::

        >>> from gaps.image_helpers import flatten_image
        >>> flat_image = flatten_image(image, 32)

    */
    int rows = image.rows / piece_size; 
    int cols = image.cols / piece_size;

    vector<Mat> piecesArray;
    // Crop pieces from original image
    for(size_t i=0;i<rows;i++){
        for (size_t j = 0; j < cols; j++)
        {
            int top = i * piece_size;
            int left = j * piece_size;
            int bottom = (i + 1) * piece_size;
            int right = (j + 1) * piece_size;
            Mat piece = image(Range(top,bottom),Range(left,right));
            piecesArray.push_back(piece);
        }
    }

    return piecesArray;
}

Mat assemble_image(vector<Mat>& pieces,int idSeq[], int rows, int cols){
    /*
    Assembles image from pieces.

    Given an array of pieces and desired image dimensions, function assembles
    image by stacking pieces.

    :params pieces:  Image pieces as an array.
    :params rows:    Number of rows in resulting image.
    :params columns: Number of columns in resulting image.

    Usage::

        >>> from gaps.image_helpers import assemble_image
        >>> from gaps.image_helpers import flatten_image
        >>> pieces, rows, cols = flatten_image(...)
        >>> original_img = assemble_image(pieces, rows, cols)

    */
    Mat res;
    for(size_t i=0;i<rows;i++){
        Mat mergedCols;
        for (size_t j = 0; j < cols; j++)
        {
            if(mergedCols.empty()) mergedCols=pieces[ idSeq[i*cols+j] ];
            else mergedCols = mergeCols(mergedCols,pieces[ idSeq[i*cols+j] ]);
        }
        if(res.empty()) res=mergedCols;
        else res = mergeRows(res,mergedCols);
    }
    return res;
}
    

float dissimilarity_measure(Mat& first_piece,Mat& second_piece,char orientation='L'){
    /*Calculates color difference over all neighboring pixels over all color channels.

    The dissimilarity measure relies on the premise that adjacent jigsaw pieces
    in the original image tend to share similar colors along their abutting
    edges, i.e., the sum (over all neighboring pixels) of squared color
    differences (over all three color bands) should be minimal. Let pieces pi ,
    pj be represented in normalized L*a*b* space by corresponding W x W x 3
    matrices, where W is the height/width of each piece (in pixels).

    :params first_piece:  First input piece for calculation.
    :params second_piece: Second input piece for calculation.
    :params orientation:  How input pieces are oriented.

                          LR => 'Left - Right'
                          TD => 'Top - Down'

    Usage::

        >>> from gaps.fitness import dissimilarity_measure
        >>> from gaps.piece import Piece
        >>> p1, p2 = Piece(), Piece()
        >>> dissimilarity_measure(p1, p2, orientation="TD")

    */

    int rows = first_piece.rows;
    int cols = first_piece.cols;
    Mat color_difference;

    //  | L | - | R |
    if (orientation == 'L'){
        color_difference = ( first_piece.col(cols-1) - second_piece.col(0) );
        color_difference.convertTo(color_difference,CV_32F);
        color_difference=color_difference / 255.0;
    }

    //  | T |
    //    |
    //  | D |
    if (orientation == 'T'){
        color_difference = ( first_piece.row(rows-1) - second_piece.row(0) );
        color_difference.convertTo(color_difference,CV_32F);
        color_difference=color_difference / 255.0;
    }

    float value = norm(color_difference,cv::NORM_L2);

    return value;
}
    
bool cmp_value(const pair<int,float>& lhs,const pair<int,float>& rhs)
{
	// if(lhs.second == rhs.second)
	// {
	// 	return lhs.first<rhs.first;
	// }
	return lhs.second < rhs.second;
}

void analyze_image(int len,
    vector<Mat>& pieces,
    float** dissimilarity_measures_LR,
    float** dissimilarity_measures_TD,
    vector<vector<vector<pair<int,float>>>>& best_match_table
){
    function<void(int,int)> update_best_match_table=[&](int first, int second){//更新最优匹配表
        //左右方向
        float measure = dissimilarity_measure(pieces[first], pieces[second], 'L');
        dissimilarity_measures_LR[first][second] = measure;
        best_match_table[0][second].push_back( make_pair(first, measure) );
        best_match_table[1][first].push_back( make_pair(second, measure) );
        //上下方向
        measure = dissimilarity_measure(pieces[first], pieces[second], 'T');
        dissimilarity_measures_TD[first][second] = measure;
        best_match_table[2][second].push_back( make_pair(first, measure) );
        best_match_table[3][first].push_back( make_pair(second, measure) );
    };
    //Calculate dissimilarity measures and best matches for each piece.
    for(int second=1;second<pieces.size();second++){
        for(int first=0;first<second;first++){
            // Left0 Right1 Top2 Down3
            update_best_match_table(first, second);
            update_best_match_table(second, first);            
        }
    }
}

// int main(int agrc, char** agrv){
//     Mat img=imread("./puzzle.jpg",1);
//     int piece_size = 48;
//     int rows = img.rows / piece_size; 
//     int cols = img.cols / piece_size;

//     printf("%d\n",img.at<uint8_t>(0,0,0));
//     Mat img1;
//     img.convertTo(img1,CV_32F);
//     cout<<img1.at<float>(0,0,0)<<endl;
//     vector<Mat> pieces = flatten_image(img,48);
//     Mat mergeImg = assemble_image(pieces,rows,cols);
    
    
//     imshow("mergeImg",mergeImg);
//     waitKey(0);
//     return 0;
// }