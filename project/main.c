#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif
#include "./algorithm.h"
#include "./frame.h"
#include "./detectLine.h"
#include "./algorithm.h"
#include "./markingLine.h"
#include <time.h>
#include <stdio.h>
EXPORT void RBF(char* path)
{

    clock_t start=clock();
    PixelPoint* targetImg=NULL;
    PixelImage* image=NULL;
    RemoveBorderFunc(path,&targetImg,&image);

    FreePoint(&targetImg);
    FreeImage(&image);

    clock_t end=clock();
    printf("./main.c/RemoveBorderFunc:\n-----c program RemoveBorderFunc useing time is  %lfs -----\n",(double) (end-start)/CLOCKS_PER_SEC);
}
EXPORT void RLF(char* path)
{
    clock_t start=clock();

    PixelPoint* targetImg=NULL;
    PixelImage* image=NULL;
    RemoveBorderFunc(path,&targetImg,&image);
    RemoveLineHandle(targetImg);

    FreeImage(&image);
    FreePoint(&targetImg);

    clock_t end=clock();
    
    printf("./main.c/RemoveLineFunc:\n-----c program RemoveLineFunc useing time is  %lfs -----\n",(double) (end-start)/CLOCKS_PER_SEC);
}
EXPORT void SLF(char* path)
{
    clock_t start=clock();
    PixelPoint* targetImg=NULL;
    PixelImage* image=NULL;
    LineType line;
    RemoveBorderFunc(path,&targetImg,&image);
    LineHandle(targetImg,&line);

    FreeImage(&image);
    FreePoint(&targetImg);
    FreeNode(line.horLine_);
    FreeNode(line.verLine_);
    clock_t end=clock();
    printf("./main.c/ShowLineFunc:\n-----c program ShowLineFunc useing time is  %lfs -----\n",(double) (end-start)/CLOCKS_PER_SEC);
}
// EXPORT void FML(char* path)
// {
//     clock_t start=clock();
//     PixelPoint* targetImg=NULL;
//     PixelImage* image=NULL;
//     LineType line;
//     RemoveBorderFunc(path,&targetImg,&image);
//     LineHandle(targetImg,&line);
//     RemoveMarkLine(&line,image);
// }

int main()
{
    SLF("../photo/527image12.jpg");
    return 0;
}  