#include "./detectLine.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
Line* InitLineHead(void)
{
    Line* L=(Line*)malloc(sizeof(Line));
    L->type_='l';
    L->length_=0;
    L->next_=NULL;
    L->pre_=NULL;
}
void CreateNode(Coordinate start,Coordinate end,char type,uint16_t length,Line* L)
{
    if(L==NULL)return;
    Line* node=(Line*)malloc(sizeof(Line));
    node->end_=end;
    node->start_=start;
    node->length_=length;
    node->type_=type;
    if (L->next_ == NULL)
    {
        node->next_=L->next_;
        node->pre_=L;
        L->next_=node;
    }
    else
    {
        node->next_=L->next_;
        node->pre_=L;
        L->next_->pre_=node;
        L->next_=node;
    }
    L->length_++;
}
void FreeList(Line* L)
{
    if (L == NULL) return;

    Line* current = L->next_;
    Line* temp;

    while (current != NULL)
    {
        temp = current;
        current = current->next_;
        free(temp);
    }

    L->next_ = NULL; 
    L->length_ = 0;  
}
void FreeNode(Line* L)
{
    if (L == NULL) return;  // 防止空指针

    Line* current = L->next_;  // 从第一个节点开始
    while (current != NULL) 
    {
        Line* temp = current;  // 保存当前节点
        current = current->next_;  // 移动到下一个节点
        free(temp);  // 释放当前节点的内存
    }

    L->next_ = NULL;  // 清空头节点的 next 指针
    L->length_ = 0;  // 重置链表长度
    free(L);
}
int compareX(const void* a, const void* b)
{
    const Coordinate* pa = (const Coordinate*)a;
    const Coordinate* pb = (const Coordinate*)b;

    if (pa->x_ < pb->x_) return -1;
    if (pa->x_ > pb->x_) return 1;
    if (pa->y_ < pb->y_) return -1;
    if (pa->y_ > pb->y_) return 1;
    return 0;
}

void SortX(PixelPoint* TargetImg)
{
    if (TargetImg == NULL || TargetImg->co_ == NULL || TargetImg->size_ == 0) {
        return;
    }
    qsort(TargetImg->co_, TargetImg->size_, sizeof(Coordinate), compareX);
}

int compareY(const void* a, const void* b)
{
    const Coordinate* pa = (const Coordinate*)a;
    const Coordinate* pb = (const Coordinate*)b;

    if (pa->y_ != pb->y_)
        return pa->y_ - pb->y_;
    return pa->x_ - pb->x_;
}

void SortY(PixelPoint* TargetImg)
{
    if (TargetImg == NULL || TargetImg->co_ == NULL || TargetImg->size_ == 0)
        return; // 防止空指针或无数据时排序崩溃

    qsort(TargetImg->co_, TargetImg->size_, sizeof(Coordinate), compareY);
}


uint8_t isVerticalRepeatLine(Line* line1,Line* line2)
{
    Line *longline,*shortLine;
    uint8_t flag=line1->length_>=line2->length_ ? 1:0;
    if(flag)
    {
        longline=line1;
        shortLine=line2;
    }
    else
    {
        longline=line2;
        shortLine=line1;
    }
    if(shortLine->start_.y_>=longline->start_.y_ && shortLine->start_.y_<=longline->end_.y_)return 1;
    if(shortLine->end_.y_>=longline->start_.y_ && shortLine->end_.y_<=longline->end_.y_) return 1;
    return 0;
}
uint8_t isHorizontalRepeatLine(Line* line1,Line* line2)
{
    Line *longline,*shortLine;
    uint8_t flag=line1->length_>=line2->length_ ? 1:0;
    if(flag)
    {
        longline=line1;
        shortLine=line2;
    }
    else
    {
        longline=line2;
        shortLine=line1;
    }
    if(shortLine->start_.x_>=longline->start_.x_ && shortLine->start_.x_<=longline->end_.x_)return 1;
    if(shortLine->end_.x_>=longline->start_.x_ && shortLine->end_.x_<=longline->end_.x_) return 1;
    return 0;
}
void RemoveLines(Coordinate* Co, uint16_t deleteNum,uint16_t start)
{
    uint16_t num=start+deleteNum;
    for(int i=start;i<num;i++)
    {
        Co[i].x_=0;
        Co[i].y_=0;
    }
}
void DeleteVerticalRepeatLine(Line* L)
{
    if(L->next_ == NULL) return; 
    Line* current = L->next_;
    Line* compare = NULL;
    while(current)
    {
        compare = current->next_;
        while(compare)
        {
            Line* nextCompare = compare->next_; 

            if(current->start_.x_ == compare->start_.x_ + 1 && isVerticalRepeatLine(current, compare))
            {
                if(current->length_ >= compare->length_)
                {
                    if(compare->pre_) compare->pre_->next_ = compare->next_;
                    if(compare->next_) compare->next_->pre_ = compare->pre_;
                    free(compare);
                }
                else
                {
                    Line* temp=current->next_;
                    if(current->pre_) current->pre_->next_ = current->next_;
                    if(current->next_) current->next_->pre_ = current->pre_;
                    free(current);
                    current = compare; 
                    current = temp;
                    nextCompare=current->next_;
                }
                L->length_--;
            }
            compare = nextCompare;
        }
        current = current->next_;
    }
}
void DeleteHorizontalRepeatLine(Line* L)
{
    if(L->next_ == NULL) return; 
    Line* current = L->next_;
    Line* compare = NULL;
    while(current)
    {
        compare = current->next_;
        while(compare)
        {
            Line* nextCompare = compare->next_; // 预存下一个节点，防止删除后指针丢失

            if(current->start_.y_==(compare->start_.y_+1) && isHorizontalRepeatLine(current, compare))
            {
                if(current->length_ > compare->length_)
                {
                    if(compare->pre_) compare->pre_->next_ = compare->next_;
                    if(compare->next_) compare->next_->pre_ = compare->pre_;
                    free(compare);
                    
                }
                else
                {
                    Line* temp=current->next_;
                    if(current->pre_) current->pre_->next_ = current->next_;
                    if(current->next_) current->next_->pre_ = current->pre_;
                    free(current);
                    current = compare; 
                    current = temp;
                    nextCompare=current->next_;
                }
                L->length_--;
            }
            compare = nextCompare;
        }
        current = current->next_;
    }
}
void FindHorizontalLine(PixelPoint* targetImg,Line* horizontalLine)
{
    int count = 0,length=0;
    char lineSymbol='\0';
    for (int i = 0; i < (int)targetImg->size_ - 1; i++)
    {
        if(targetImg->co_[i].y_ == targetImg->co_[i + 1].y_ && targetImg->co_[i].x_ >= targetImg->co_[i + 1].x_ - 1)
        {
            count++;
        }
        else
        {
            if(count > 30)
            {
                length=targetImg->co_[i].x_-targetImg->co_[i - count].x_;
                if(length==count)
                {
                    lineSymbol='l';
                }
                else
                {
                    lineSymbol='v';
                }
                CreateNode(targetImg->co_[i - count], targetImg->co_[i], lineSymbol, length, horizontalLine);
            }
            count = 0;
        }
    }
}
void FindVerticalLine(PixelPoint* target,Line* verticalLine)
{
    int count = 0,length=0;
    char lineSymbol='0';
    for (int i = 0; i < (int) target->size_ - 1; i++)
    {
        if(target->co_[i].x_ == target->co_[i + 1].x_ && target->co_[i].y_ >= target->co_[i + 1].y_ - 1)
        {
            count++;
        }
        else
        {
            if(count > 30)
            {

                length=target->co_[i].y_-target->co_[i - count].y_;
                if(length==count)
                {
                    lineSymbol='l';
                }
                else
                {
                    lineSymbol='v';
                }
                CreateNode(target->co_[i - count], target->co_[i], lineSymbol, length, verticalLine);
            }
            count = 0;
        }
    }
}
void WriteStraigLine(Line* Vline,Line* Hline)
{
    Line *vL=Vline->next_;
    Line *hL=Hline->next_;
    FILE *file = fopen("./StraightLine.txt", "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }
    for (int i = 0;i<Vline->length_; i++)
    {
        
        for(int j=0;j<vL->length_;j++)
        {
            fprintf(file, "%d,%d\n",vL->start_.x_,vL->start_.y_+j); 
        }
        vL=vL->next_;
    }
    for (int i = 0;i<Hline->length_; i++)
    {
        
        for(int j=0;j<hL->length_;j++)
        {
            fprintf(file, "%d,%d\n",hL->start_.x_+j,hL->start_.y_);
        }
        hL=hL->next_;
    }
    fclose(file);
}
void WriteLsp(Line* v,Line* h)
{
    FILE *file = fopen("./line.txt", "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }
    while(v!=NULL)
    {
        //fprintf(file,"%d,%d,%d,%d\n",v->start_.x_,3000-v->start_.y_,v->end_.x_,3000-v->end_.y_); //y轴反转
        fprintf(file,"%d,%d,%d,%d\n",v->start_.x_,v->start_.y_,v->end_.x_,v->end_.y_);
        v=v->next_;
    }
    while(h!=NULL)
    {
        //fprintf(file,"%d,%d,%d,%d\n",h->start_.x_,3000-h->start_.y_,h->end_.x_,3000-h->end_.y_); //y轴反转
        fprintf(file,"%d,%d,%d,%d\n",h->start_.x_,h->start_.y_,h->end_.x_,h->end_.y_);
        h=h->next_;
    }
    fclose(file);
} 
void DeleteVirtualLine(Line* head)
{
    Line* node = head;
    while (node) {
        Line* nextNode = node->next_; // 先保存下一个节点
        
        if (node->type_ == 'v') { // 找到要删除的节点
            if (node->pre_) {
                node->pre_->next_ = node->next_;
            }
            if (node->next_) {
                node->next_->pre_ = node->pre_;
            }
            free(node); // 释放当前节点
            head->length_--;
        }
        
        node = nextNode; // 继续遍历
    }
}
void BreakVerLine(Line* verLine,Line* headHorLine,Line* breakLinehead)
{
    Line* line=headHorLine->next_;
    PixelPoint* p;
    uint16_t length=0; 
    CreatePoint(&p,20);
    p->co_[p->size_]=verLine->end_;
    p->size_++;
    while(line!=NULL)
    {
        if(line->start_.x_<verLine->start_.x_ && line->end_.x_>verLine->start_.x_ 
            && line->start_.y_>verLine->start_.y_ && line->end_.y_<verLine->end_.y_)
        {
            Coordinate temp={0};
            temp.x_=verLine->start_.x_;
            temp.y_=line->start_.y_;
            p->co_[p->size_]=temp;
            p->size_++;
        }
        line=line->next_;
    }
    p->co_[p->size_]=verLine->start_;
    for(int i=0;i<p->size_;i++)
    {
        length=p->co_[i].y_-p->co_[i+1].y_;
        if(length>10)CreateNode(p->co_[i],p->co_[i+1],'\0',length,breakLinehead);
    }
    FreePoint(&p);
}




void BreakHorLine(Line* horLine,Line* headverLine,Line* breakVerLinehead)
{
    Line* line=headverLine->next_;
    PixelPoint* p;
    uint16_t length=0; 
    CreatePoint(&p,20);
    p->co_[p->size_]=horLine->end_;
    p->size_++;
    while(line!=NULL)
    {
        if(line->start_.x_>horLine->start_.x_ && line->end_.x_<horLine->end_.x_ 
            && line->start_.y_<horLine->start_.y_ && line->end_.y_>horLine->end_.y_)
        {
            Coordinate temp={0};
            temp.x_=line->start_.x_;
            temp.y_=horLine->start_.y_;
            p->co_[p->size_]=temp;
            p->size_++;
        }
        line=line->next_;
    }
    p->co_[p->size_]=horLine->start_;
    for(int i=0;i<p->size_;i++)
    {
        length=p->co_[i].x_-p->co_[i+1].x_;
        if(length>10)CreateNode(p->co_[i],p->co_[i+1],'\0',length,breakVerLinehead);
    }
    FreePoint(&p); 
}
void BreakLineHandle(Line* verLineHead,Line* horLineHead)
{
    Line* breakHorLinehead=InitLineHead();
    Line* breakVerLinehead=InitLineHead();
    Line* l1=NULL;
    l1=verLineHead->next_;;
    for(int i=0;i<verLineHead->length_;i++)
    {
        BreakVerLine(l1,horLineHead,breakHorLinehead);
        l1=l1->next_;
    }
    l1=horLineHead->next_;
    for(int i=0;i<horLineHead->length_;i++)
    {
        BreakHorLine(l1,verLineHead,breakVerLinehead);
        l1=l1->next_;
    }
    WriteLsp(breakVerLinehead->next_,breakHorLinehead->next_);
    FreeNode(breakHorLinehead);
    FreeNode(breakVerLinehead);
}
void LineHandle(PixelPoint* targetImg,LineType* line)
{
    Line* verticalLine=InitLineHead();
    SortX(targetImg);                      //x从小到大,y值在每个x中从小到大

    
    FindVerticalLine(targetImg,verticalLine);

    Line* l=verticalLine->next_;

    SortY(targetImg);                       //（在x已经有序的情况下(调用SortX)调用SortY函数）x从小到大 y值在每个x中从小到大

    Line* horizontalLine=InitLineHead();
    FindHorizontalLine(targetImg,horizontalLine);

    
    //Line* l2=horizontalLine->next_;

    // //DeleteVirtualLine(horizontalLine);
    DeleteHorizontalRepeatLine(horizontalLine);
    // //DeleteVirtualLine(verticalLine);
    DeleteVerticalRepeatLine(verticalLine);
    //WriteLsp(verticalLine->next_,horizontalLine->next_);




    
    
    //WriteStraigLine(verticalLine,horizontalLine);
   

    BreakLineHandle(verticalLine,horizontalLine);





    printf("./detectLine.c/StraigLineHandle:\n");
    printf("------verticalLine line num is  %d\n", verticalLine->length_);
    printf("------horizontalLine line num is %d\n", horizontalLine->length_);
    line->horLine_=horizontalLine;
    line->verLine_=verticalLine;
}
void RemoveAllLine(PixelPoint* targetImg)
{
    SortX(targetImg);
    int count = 0;
    for (int i = 0; i < (int)targetImg->size_ - 1; i++)
    {
        if(targetImg->co_[i].x_ == targetImg->co_[i + 1].x_ && targetImg->co_[i].y_ == targetImg->co_[i + 1].y_ - 1)
        {
            count++;
        }
        else
        {
            if(count > 30)
            {
                RemoveLines(targetImg->co_,count,i-count);
            }
            count = 0;
        }
    }
    SortY(targetImg);
    for(int i=0;i<(int)targetImg->size_;i++)
    {
        if(targetImg->co_[i].y_ == targetImg->co_[i + 1].y_ && targetImg->co_[i].x_ == targetImg->co_[i + 1].x_ - 1)
        {
            count++;
            if(targetImg->co_[i].x_==0 && targetImg->co_[i].y_==0)return;
        }
        else
        {
            if(count > 30)
            {
                
                RemoveLines(targetImg->co_,count,i-count);
            }
            count = 0;
        }
    }
}
void RemoveLineHandle(PixelPoint* targetImg)
{
    uint16_t count=0;
    RemoveAllLine(targetImg);
    
    FILE *file = fopen("./RemoveLine.txt", "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }
    for (int x = 0;x<targetImg->size_; x++)
    {
        if(targetImg->co_[x].x_!=0 && targetImg->co_[x].y_!=0)
        {
            fprintf(file, "%d,%d\n",targetImg->co_[x].x_,targetImg->co_[x].y_);
        }
    }
    fclose(file);
}
