function [ OFFSET_R_ALONG_x, OFFSET_R_ALONG_y, OFFSET_G_ALONG_x, OFFSET_G_ALONG_y, SSD_Image] = im_align1(r, g, b)
  
  OFFSET = 30;
  
  # R G B Size
  
  [Px, Py] = size(b);
  
  # New Borders
  NEW_B = b(OFFSET : Px - OFFSET, OFFSET : Py - OFFSET);
  
  [Offset_x,Offset_y] = size(NEW_B);
  
  # Maximizing SSD
  
  SSD_G = 255 * Offset_x * Offset_y;
  
  SSD_R = 255 * Offset_x * Offset_y;
  
  for i = -15:15
    for j = -15:15
      
      NEW_R = r(OFFSET + i : Px - OFFSET + i , OFFSET + j : Py - OFFSET + j);
      
      NEW_G = g(OFFSET + i : Px - OFFSET + i , OFFSET + j : Py - OFFSET + j);
         
      ## Finding next power of 2
      OffsetR = (NEW_B - NEW_R).^2;
      
      OffsetG = (NEW_B - NEW_G).^2;
      
      temp_SSD_R = sum( sum(OffsetR) );
      
      temp_SSD_G = sum( sum(OffsetG) );
      
      if temp_SSD_R <= SSD_R
        SSD_R = temp_SSD_R;
        OFFSET_R_ALONG_x = i;
        OFFSET_R_ALONG_y = j;
      endif


      if temp_SSD_G <= SSD_G
        SSD_G = temp_SSD_G; 
        OFFSET_G_ALONG_x = i; 
        OFFSET_G_ALONG_y = j;
      endif
       
    endfor
  endfor
 
 
  # Calculating new R and G values using SSDs
  NEW_G = g(OFFSET + OFFSET_G_ALONG_x : Px - OFFSET + OFFSET_G_ALONG_x , OFFSET + OFFSET_G_ALONG_y : Py - OFFSET + OFFSET_G_ALONG_y);
  
  NEW_R = r(OFFSET + OFFSET_R_ALONG_x : Px - OFFSET + OFFSET_R_ALONG_x , OFFSET + OFFSET_R_ALONG_y : Py - OFFSET + OFFSET_R_ALONG_y);
  
  SSD_Image = cat(3, NEW_R, NEW_G, NEW_B);
endfunction