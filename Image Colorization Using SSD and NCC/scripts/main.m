clc
close all
clear all

pkg load image

Input_Imgaes = 1;

while (Input_Imgaes < 7)
  
  IMG = imread(strcat('./../Input Images/image',int2str(Input_Imgaes),'.jpg'));

  [x,y] = size(IMG);
  
  Red = IMG(2 * floor(x/3) : x-1 , 1 : y);
  
  Green = IMG(floor(x/3) : 2 * floor(x/3) , 1 : y);
  
  Blue = IMG(1 : floor(x/3) + 1 , 1 : y);
  
  # Merging the R G B planes into one image
  COLORING = cat(3, Red, Green, Blue);
  imwrite(COLORING, strcat('image',int2str(Input_Imgaes),'-color.jpg'));  
  
  
  # Calling SSD Alignment
  
  [ OFFSET_R_ALONG_x, OFFSET_R_ALONG_y, OFFSET_G_ALONG_x, OFFSET_G_ALONG_y, SSD_Image] = im_align1(Red, Green, Blue);

  disp(strcat('Image',int2str(Input_Imgaes)));
  
  disp(strcat('R alignment shift using SSD: [',int2str(OFFSET_R_ALONG_x),',',int2str(OFFSET_R_ALONG_y),']'));
  
  disp(strcat('G alignment shift using SSD: [',int2str(OFFSET_G_ALONG_x),',',int2str(OFFSET_G_ALONG_y),']'));
  
  imwrite(SSD_Image, strcat('./../Output Images/image', int2str(Input_Imgaes), '-ssd.jpg'));
  

  # Calling NCC Alignment
  
  [OFFSET_R_ALONG_x, OFFSET_R_ALONG_y, OFFSET_G_ALONG_x, OFFSET_G_ALONG_y, NCC_Image]= im_align2(Red, Green, Blue);
  
  disp(strcat('R alignment shift using NCC: [',int2str(OFFSET_R_ALONG_x), ',', int2str(OFFSET_R_ALONG_y),']'));
  
  disp(strcat('G alignment shift using NCC: [',int2str(OFFSET_G_ALONG_x), ',', int2str(OFFSET_G_ALONG_y),']'));
  
  imwrite(NCC_Image, strcat('./../Output Images/image', int2str(Input_Imgaes), '-ncc.jpg'));
  
  Input_Imgaes = Input_Imgaes + 1;

end
