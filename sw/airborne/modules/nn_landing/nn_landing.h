/*
 * Copyright (C) Huizerd
 *
 * This file is part of paparazzi
 *
 * paparazzi is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * paparazzi is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with paparazzi; see the file COPYING.  If not, see
 * <http://www.gnu.org/licenses/>.
 */
/**
 * @file "modules/nn_landing/nn_landing.h"
 * @author Huizerd
 * Artificial neural networks for optical flow landing.
 */

#pragma once

// C standard library headers
#include <stdbool.h>
#include <stdint.h>

// Module functions
extern void nn_landing_init(void);
extern void nn_landing_event(void);

// Divergence + derivative and thrust for logging
// And recording variable to easily identify descents
// TODO: is extern here dangerous?
extern float divergence, divergence_dot, acc_lp, thrust_raw, thrust, thrust_lp;
extern float acceleration_sp;
extern float div_gt, divdot_gt;
extern uint16_t spike_count;
extern uint8_t record;

// Struct to hold settings
struct NNLandingSettings {
  float thrust_effect;          ///< thrust effectiveness
  float thrust_p_gain;          ///< P-gain for active thrust control
  float thrust_i_gain;          ///< I-gain for active thrust control
};

extern struct NNLandingSettings nl_settings;