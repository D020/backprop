#pragma once
class Plot {
  public:
	Plot();
	Plot(int width,int height);
	
	int getWidth();
	int getHeight();

	void save(const char* path);
	void plot(int x, int y, float r, float g, float b);

	int width, height;
	float* buf;
};
