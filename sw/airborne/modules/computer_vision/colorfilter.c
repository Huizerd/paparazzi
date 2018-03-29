/*
 * Copyright (C) 2015
 *
 * This file is part of Paparazzi.
 *
 * Paparazzi is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * Paparazzi is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Paparazzi; see the file COPYING.  If not, write to
 * the Free Software Foundation, 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/**
 * @file modules/computer_vision/colorfilter.c
 */

// Own header
#include "modules/computer_vision/colorfilter.h"
#include <stdio.h>

#include "modules/computer_vision/lib/vision/image.h"

#ifndef COLORFILTER_FPS
#define COLORFILTER_FPS 0       ///< Default FPS (zero means run at camera fps)
#endif
PRINT_CONFIG_VAR(COLORFILTER_FPS)


#ifndef COLORFILTER_SEND_OBSTACLE
#define COLORFILTER_SEND_OBSTACLE FALSE    ///< Default sonar/agl to use in opticflow visual_estimator
#endif
PRINT_CONFIG_VAR(COLORFILTER_SEND_OBSTACLE)

struct video_listener *listener = NULL;

// Filter Settings David
uint8_t color_lum_min = 71;//105;
uint8_t color_lum_max = 130;//205;
uint8_t color_cb_min  = 59;//52;
uint8_t color_cb_max  = 93;//140;
uint8_t color_cr_min  = 63;//180;
uint8_t color_cr_max  = 105;//255;

// Result
uint16_t color_count = 0;
uint16_t color_count_boxes[VER_SUBBOXES][HOR_SUBBOXES] = {0};

#include "subsystems/abi.h"
#include "colorfilter.h"

uint16_t ctr=0;
uint16_t *count_p_r=&ctr;
uint16_t ctl=0;
uint16_t *count_p_l=&ctl;

// Function
struct image_t *colorfilter_func(struct image_t *img)
{
  // Filter
  color_count = image_yuv422_colorfilt_box(img, img,
                                       color_lum_min, color_lum_max,
                                       color_cb_min, color_cb_max,
                                       color_cr_min, color_cr_max, &ctr, &ctl
                                      );

  color_count_boxes = image_yuv422_colorfilt_multibox(img, img,
                                                      VER_SUBBOXES, HOR_SUBBOXES,
                                                      color_count_boxes, origin_box,
                                                      h_box, w_box,
                                                      color_lum_min, color_lum_max,
                                                      color_cb_min, color_cb_max,
                                                      color_cr_min, color_cr_max
  );
  for (int i_print = 0; i_print < VER_SUBBOXES; i_print++) {
    for (int j_print = 0; j_print < HOR_SUBBOXES; j_print++) {

      printf("Box %d: %d", i_print*VER_SUBBOXES + j_print, color_count_boxes[i_print][j_print]);
    }
  }
  //printf("Count right: %d", *count_p_r);

  if (COLORFILTER_SEND_OBSTACLE) {
    if (color_count > 20)
    {
      AbiSendMsgOBSTACLE_DETECTION(OBS_DETECTION_COLOR_ID, 1.f, 0.f, 0.f);
    }
    else
    {
      AbiSendMsgOBSTACLE_DETECTION(OBS_DETECTION_COLOR_ID, 10.f, 0.f, 0.f);
    }
  }

  return img; // Colorfilter did not make a new image
}

void colorfilter_init(void)
{
  listener = cv_add_to_device(&COLORFILTER_CAMERA, colorfilter_func, COLORFILTER_FPS);
}
