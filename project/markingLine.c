#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "./algorithm.h"
typedef struct Arrow
{ 
    uint16_t x_;
    uint16_t y_;
    uint8_t width;
}Arrow;
uint8_t FindLineWidth(Arrow* arrow, uint16_t length)
{
    uint8_t count[256] = {0};
    uint8_t maxWidth = 0;      
    uint8_t maxCount = 0;      
    for (uint16_t i = 0; i < length; i++)
    {
        count[arrow[i].width]++; 
        if (count[arrow[i].width] > maxCount)
        {
            maxCount = count[arrow[i].width];
            maxWidth = arrow[i].width;
        }
    }
    return maxWidth;
}
uint8_t SearchHorPixel(PixelImage* img,Line line)
{
    uint8_t temp=0,arrowNum=0;
    for(int i=0;i<line.length_;i++,line.start_.x_++)
    {
        if(img->image_[line.start_.x_][line.start_.y_+1]==0 && img->image_[line.start_.x_][line.start_.y_-1]==0)
        {
            if(++temp>6)
            {
                arrowNum++;
                temp=0;
            }
        }
        else
        {
            temp=0;
        }
    }
    return arrowNum;
    
}
void FindHorArrow(Line* horLineHead,PixelImage* img)
{
    uint8_t arrawNum=0;
    Line* line=horLineHead->next_;
    for(int i=0;i<horLineHead->length_;i++)
    {
        arrawNum=SearchHorPixel(img,*line);
        if(arrawNum>0)
        {
            line->type_='M';
            //printf("%u,%u,%u,%u,%u\n",line->start_.x_,line->start_.y_,line->end_.x_,line->end_.y_,arrawNum);
        }
        line=line->next_;
    }
}

uint8_t Func(Arrow* temp,uint16_t length,uint8_t num)
{
    
    uint8_t j=0;
    uint8_t arrowNum=0;
    uint16_t flag=0;
    for(int i=0;i<length;i++)
    {
        if(temp[i].width>num)
        {
            j=0;
            while(temp[i+j].width>num)
            {
                j++;
            }
            if(j>7 && j<13)
            {
                arrowNum++;
                //printf("arrow:%d,%d,%d,%d,%d\n",num,temp[i].x_,temp[i].y_,temp[i+j-1].x_,temp[i+j-1].y_);
            }   
            i=i+j-1;
        }
    }
    //printf("%d\n",arrowNum);
    return arrowNum; 
}
uint8_t SearchVerPixel(PixelImage* img,Line line)
{
    uint8_t left=0,right=0,exitWhileFlag=0;
    uint16_t a=0;
    Arrow* temp=(Arrow*)malloc(sizeof(Arrow)*line.length_);
    while(a<line.length_)
    {
        left=0;
        right=0;
        while(img->image_[line.start_.x_+right+1][line.start_.y_+a]==0)
        {
            right++;
            if(right>10)
            {
                exitWhileFlag=1;
                right=0;
                left=0;
                break;
            }
        }
        while(img->image_[line.start_.x_-left-1][line.start_.y_+a]==0)
        {
            if(exitWhileFlag==1)break;
            left++;
            if(left>10)
            {
                right=0;
                left=0;
                break;
            }
        }
        exitWhileFlag=0;
        temp[a].x_=line.start_.x_;
        temp[a].y_=line.start_.y_+a;
        temp[a].width=left+right+1;
        a++;
    }
    // for(int i=0;i<line.length_;i++)
    // {
    //     printf("%d\n",temp[i].width);
    // }
    uint8_t b=FindLineWidth(temp,line.length_);
    uint8_t arrowNUm=Func(temp,line.length_,b);
    

    free(temp);
    //printf("line width is %d\n",b);
    return arrowNUm;
}
void FindVerArrow(Line* verLineHead,PixelImage* img)
{
    uint8_t arrawNum=0;
    Line* line=verLineHead->next_;
    for(int i=0;i<verLineHead->length_;i++)
    {
        arrawNum=SearchVerPixel(img,*line);
        if(arrawNum>0)
        {
            line->type_='M';
            //printf("(%u,%u,%u,%u)\n",line->start_.x_,line->start_.y_,line->end_.x_,line->end_.y_);
            //return ;
        }
        line=line->next_;
    }
}

void RemoveMarkLine(LineType* line,PixelImage* img)
{
    Line* verLineHead=line->verLine_;
    Line* horLineHead=line->horLine_;
    FindHorArrow(horLineHead,img);
    FindVerArrow(verLineHead,img);
}