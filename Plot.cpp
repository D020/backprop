#include "Plot.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits>
#include <iostream>
#include <algorithm>
#include <math.h>
Plot::Plot(){
	width  = 0;
	height = 0;
	buf    = 0;
}

Plot::Plot(int width, int height){
	this->width  = width;
	this->height = height;
	buf = (float*) calloc(width*height*3,sizeof(float));
}

int Plot::getWidth(){
	return width;
}

int Plot::getHeight(){
	return height;
}

void Plot::plot(int x, int y, float r, float g, float b){
	bool insideWidth  = (0<=x && x<width);
	bool insideHeight = (0<=y && y<height);
	if(insideWidth && insideHeight){
		unsigned int index = (y*width + x)*3;
		buf[index+0] = r;
		buf[index+1] = g;
		buf[index+2] = b;
	}
	else
		printf("Invalid plot coordinates\n");
}

float clamp(float val, float a, float b){
	if(val<a)
		val = a;
	else if(b<val)
		val = b;
	return val;
}

void Plot::save(const char* path){
	FILE* fp;
	fopen_s(&fp, path,"wb");
	fprintf(fp,"P6\n%d %d\n255\n",width,height);

	//float scalar (255.0/(hdr_max-hdr_min));

	//float k = 0.55;

	float a = 0;
	float b = 1.0;

	for (int i=0; i<height; i++){
		for (int j=0; j<width; j++){
			static unsigned char color[3];
			unsigned int index = (i*width + j)*3;
			//color[0] = clamp(sqrt(buf[index  ]/k),a,b);
			//color[1] = clamp(sqrt(buf[index+1]/k),a,b);
			//color[2] = clamp(sqrt(buf[index+2]/k),a,b);
            color[0] = clamp((buf[index  ]),a,b)*255.0;
			color[1] = clamp((buf[index+1]),a,b)*255.0;
			color[2] = clamp((buf[index+2]),a,b)*255.0;
			fwrite(color,1,3,fp);
		}
	}

	fclose(fp);

}