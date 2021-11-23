function [OFFSET_R_ALONG_x, OFFSET_R_ALONG_y, OFFSET_G_ALONG_x, OFFSET_G_ALONG_y, NCC_Image] = im_align2(r,g,b)
  
  OFFSET = 30;
    
  # Calaculating sizes of R G B
  [Px,Py] = size(b);
  
  # Getting rid of the borders in the channel
  NEW_B = b(OFFSET : Px - OFFSET, OFFSET : Py - OFFSET);
  
  # Making templates of the R and G channels
  Template_G = g(100:150,100:150);
  Template_R = r(100:150,100:150);
  
  # Calculating the Normalized cRoss Correlation
  cR = normxcorr2(Template_R, b);
  [Rx_Peak, Ry_Peak] = find(cR == max(cR(:)));
  
  
  cG = normxcorr2(Template_G, b);
  [Gx_Peak, Gy_Peak] = find( cG == max(cG(:)));
  
  
  # Offset of R Channel
  OFFSET_R_ALONG_x = 100 - Rx_Peak + 50;
  
  OFFSET_R_ALONG_y = 100 - Ry_Peak + 50;
  
  # Offset of G Channel
  OFFSET_G_ALONG_x = 100 - Gx_Peak + 50;
  
  OFFSET_G_ALONG_y = 100 - Gy_Peak + 50;
  
  # Calculating new R and G channel OFFSETs using the OFFSET calculated by NCC
  NCC_R = r(OFFSET + OFFSET_R_ALONG_x : Px - OFFSET + OFFSET_R_ALONG_x , OFFSET + OFFSET_R_ALONG_y : Py-OFFSET + OFFSET_R_ALONG_y);
  
  NCC_G = g(OFFSET+OFFSET_G_ALONG_x : Px - OFFSET + OFFSET_G_ALONG_x , OFFSET + OFFSET_G_ALONG_y : Py - OFFSET + OFFSET_G_ALONG_y);
  
  NCC_Image = cat(3,NCC_R,NCC_G,NEW_B);
endfunction