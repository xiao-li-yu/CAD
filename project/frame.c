#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "./stb_image.h"
#include "./stb_image_write.h"
#include "./frame.h"
#include <stdio.h>//Img,startPoint,Bound
void BFS(PixelImage* pixelImg,Coordinate* startPoint,PixelImage* bound)
{
    uint16_t queue[QUEUESIZE][2];  
    uint32_t front = 0, rear = 0;  
    uint32_t result_count=0;
    queue[rear][0] = startPoint->x_;
    queue[rear][1] = startPoint->y_;
    rear++;
    bound->image_[startPoint->x_][startPoint->y_] = 1;  
    int8_t directions[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    result_count = 0;  
    while (front < rear) {
        uint16_t x = queue[front][0];
        uint16_t y = queue[front][1];
        front++;
        result_count++;
        for (uint8_t i = 0; i < 4; i++) {
            int16_t nx = x + directions[i][0];
            int16_t ny = y + directions[i][1];
            if (nx >= 0 && ny >= 0 && nx < pixelImg->property_.x_ && ny < pixelImg->property_.y_ && !bound->image_[nx][ny] && pixelImg->image_[nx][ny] == 0)
            {
                bound->image_[nx][ny] = 1;  
                queue[rear][0] = nx;
                queue[rear][1] = ny;
                rear++;
            }
        }
    }
    printf("./frame.c/BFS:\n-----BFS of funtion use queue size is %u,set QUEUESIZE size is %u -----\n",rear,QUEUESIZE);
    if(result_count<4000)
    {
        printf("The image border may is wrong\n");
    }
}
void EdgeBoundBoxCo(PixelImage* bound,PixelPoint* edgeBound)
{
    uint16_t temp=0;
    for(uint16_t x=bound->property_.x_/2; x<bound->property_.x_; x++)
    {
        for(uint16_t y=bound->property_.y_/2;y>0;y--)
        {
            if(bound->image_[x][y]==1)
            {
                edgeBound->co_[edgeBound->size_].x_=x;
                edgeBound->co_[edgeBound->size_].y_=y;
                edgeBound->size_++;
                break;
            }
        }
        for(uint16_t y=bound->property_.y_/2+1; y<bound->property_.y_ ; y++)
        {
            
            if(bound->image_[x][y]==1)
            {
                edgeBound->co_[edgeBound->size_].x_=x;
                edgeBound->co_[edgeBound->size_].y_=y;
                edgeBound->size_++;
                break;
            }
        }
        temp=edgeBound->size_-1;
        if(edgeBound->co_[temp].y_==bound->property_.y_/2)break;
    }
    temp=0;
    for(uint16_t x=bound->property_.x_/2-1;x>0;x--)
    {
        for(uint16_t y=bound->property_.y_/2;y>0;y--)
        {
            if(bound->image_[x][y]==1)
            {
                edgeBound->co_[edgeBound->size_].x_=x;
                edgeBound->co_[edgeBound->size_].y_=y;
                edgeBound->size_++;
                break;
            }
        }
        for(uint16_t y=bound->property_.y_/2+1;y<bound->property_.y_;y++)
        {
            if(bound->image_[x][y]==1)
            {
                edgeBound->co_[edgeBound->size_].x_=x;
                edgeBound->co_[edgeBound->size_].y_=y;
                edgeBound->size_++;
                break;
            }
        }
        temp=edgeBound->size_-1;
        if(edgeBound->co_[temp].y_==bound->property_.y_/2)break;
    }
    printf("./frame.c/BoundingBoxCo:\n-----edgeBoundBoxCoSize is %u,set EDGECOORDINATESIZE size is %u -----\n",edgeBound->size_,EDGECOORDINATESIZE);
}
void TargetImage(PixelImage* pixelImg,PixelPoint* RBImg,PixelPoint* edgebound)
{
    uint16_t starty1=0,starty2=0,fixedPointx=0;
    for(int i=0;i<edgebound->size_;)
    {
        fixedPointx=edgebound->co_[i].x_;
        
        starty1=edgebound->co_[i++].y_;
        starty2=edgebound->co_[i++].y_;
        for(uint16_t y=starty1+1;y<starty2;y++)
        {
            if(pixelImg->image_[fixedPointx][y]==0)
            {
                RBImg->co_[RBImg->size_].x_=fixedPointx;
                RBImg->co_[RBImg->size_].y_=y;
                RBImg->size_++;
            }
        }
    }
    printf("./frame.c/TargetImage:\n-----Final number of pixels %u,Set FINALNUMBEROFPIXELS number is %u -----\n",RBImg->size_,RBIMG);
}
void FindStartingPointOfBFS(Coordinate* startPoint,PixelImage* pixelImg)
{

    for(int y=3;y<pixelImg->property_.y_-3;y++)
    {
        for(int x=3;x<pixelImg->property_.x_-3;x++)
        {
            if(pixelImg->image_[x][y]==0)
            {
                printf("./frame.c/FindStartingPointOfBFS:\n-----(%d,%d)is the starting point coordinates of BFS traversal-----\n",x,y);
                //printf("220");
                startPoint->x_=x;
                startPoint->y_=y;
                return;
            }
        }
    }
    printf("No coordinates found\n");

}
void BinarizeImage(uint8_t *img_data,uint16_t threshold,PixelImage* pixelimg)
{
    
    for (int y = 0; y < pixelimg->property_.y_; y++) {
        for (int x = 0; x < pixelimg->property_.x_; x++) {
            // 获取当前像素点的灰度值（假设图像是RGB，取R通道即可）
            
            int index = (y * pixelimg->property_.x_ + x) * 3;
            uint8_t r = img_data[index];  // 获取红色通道的值
            uint8_t g = img_data[index + 1];  // 获取绿色通道的值
            uint8_t b = img_data[index + 2];  // 获取蓝色通道的值

            // 计算灰度值，这里用简单的平均值法（也可以使用其他加权方法）
            uint16_t gray = (r + g + b) / 3;
            // 根据阈值将灰度值转换为二值值
            
            pixelimg->image_[x][y] = (gray >= threshold) ? 255 : 0;
            
        }
    }
}
void WriteAllFunc(PixelPoint* RBimg)
{
    FILE *file = fopen("./RemoveFrame.txt", "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }
    for (int x = 0;x<RBimg->size_; x++)
    {
        fprintf(file, "%d,%d\n",RBimg->co_[x].x_,RBimg->co_[x].y_);
    }
    fclose(file);
}
void RemoveBorderFunc(const char* imagePath,PixelPoint** targetImg,PixelImage** image)
{
    printf("image path: %s\n",imagePath);
    PixelImage *img,*bound;
    PixelPoint *edgeBound,*RBImg;
    int width=0,height=0,channels=0,threshold=180;
    
    uint8_t* imageData=stbi_load(imagePath, &width, &height, &channels, 0);
    if (imageData == NULL)
    {
        printf("Failed to load image,The image does not exist or the path is incorrect\n");
        return ;
    }
    CreateImage(&img,(uint16_t)width,(uint16_t)height);
    BinarizeImage(imageData,threshold,img);
    
    stbi_image_free(imageData);//free imageData space
    
    Coordinate* startPoint=malloc(sizeof(Coordinate));
    FindStartingPointOfBFS(startPoint,img);

    CreateImage(&bound,(uint16_t)width,(uint16_t)height);
    BFS(img,startPoint,bound);

    
    CreatePoint(&edgeBound,EDGECOORDINATESIZE);

    EdgeBoundBoxCo(bound,edgeBound);
    
    
    
    
    CreatePoint(&RBImg,RBIMG);
    TargetImage(img,RBImg,edgeBound);

    WriteAllFunc(RBImg);
    FreeImage(&bound);
    FreePoint(&edgeBound);
    free(startPoint);
    *targetImg=RBImg;
    *image=img;
}