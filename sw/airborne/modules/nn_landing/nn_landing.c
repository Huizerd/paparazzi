/*
 * Copyright (C) Huizerd
 * Copyright (C) Kirk Scheper
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
 * @file "modules/nn_landing/nn_landing.c"
 * @author Huizerd
 * Artificial neural networks for optical flow landing.
 */

// Header for this file
#include "modules/nn_landing/nn_landing.h"

// Header with network parameters
#include "modules/nn_landing/nn_weights.h"

// Header for UART communication
#include "modules/uart_driver/uart_driver.h"

// Paparazzi headers
#include "firmwares/rotorcraft/guidance/guidance_v_adapt.h"
#include "firmwares/rotorcraft/stabilization.h"
#include "generated/airframe.h"
#include "paparazzi.h"
#include "subsystems/abi.h"

// Used for automated landing
#include "autopilot.h"
#include "filters/low_pass_filter.h"
#include "subsystems/datalink/telemetry.h"

// For measuring time
#include "mcu_periph/sys_time.h"

// C standard library headers
#include <stdbool.h>
#include <stdio.h>

// Use optical flow estimates
#ifndef NL_OPTICAL_FLOW_ID
#define NL_OPTICAL_FLOW_ID ABI_BROADCAST
#endif
PRINT_CONFIG_VAR(NL_OPTICAL_FLOW_ID)

// Other default values
// Closed-loop thrust control, else linear transform
#define NL_ACTIVE_CONTROL true
// #define NL_UART_CONTROL

// Gains for closed-loop control
#ifndef NL_THRUST_EFFECT
#define NL_THRUST_EFFECT 0.01f
#endif
#ifndef NL_THRUST_P_GAIN
#define NL_THRUST_P_GAIN 2.0f
#endif
#ifndef NL_THRUST_I_GAIN
#define NL_THRUST_I_GAIN 0.3f
#endif

// Optical flow settings
#ifndef NL_OF_FILTER_CUTOFF
#define NL_OF_FILTER_CUTOFF 1.5f
#endif

// Events
static abi_event optical_flow_ev;

// Low-pass filters for acceleration and thrust
static Butterworth2LowPass accel_ned_filt;
static Butterworth2LowPass thrust_filt;

// Variables retained between module calls
// For divergence + derivative, low-passed acceleration, thrust
float divergence, divergence_dot, acc_lp, thrust, thrust_lp;
float acceleration_sp;
float div_gt, divdot_gt;
float div_gt_tmp;
// Spike count --> not used
uint16_t spike_count;
// For recording
uint8_t record;
// For control
static float nominal_throttle;
static bool active_control;

// Kirk's network uses half of our divergence
static float divergence_half, divergence_dot_half;

// Network specification
// Layers
float input_layer_out[nr_input_neurons] = {0};
float hidden_layer_out[nr_hidden_neurons] = {0};
float layer2_out[nr_output_neurons] = {0};
// Type
#if NN_TYPE == NN || NN_TYPE == RNN
static float relu(float val)
{
  BoundLower(val, 0.f);
  return val;
}

#elif NN_TYPE == CTRNN

static float sigmoid(float val)
{
  return 1.f / (1.f + expf(-val));
}
#endif

// Struct to hold settings
struct NNLandingSettings nl_settings;

// Sending stuff to ground station
// Divergence + derivative, height, velocity, acceleration, thrust, mode
static void send_nl(struct transport_tx *trans, struct link_device *dev) {
  pprz_msg_send_SPIKING_LANDING(
      trans, dev, AC_ID, &divergence, &divergence_dot,
      &(stateGetPositionNed_f()->x), &(stateGetPositionNed_f()->y),
      &(stateGetPositionNed_f()->z), &(stateGetPositionEnu_f()->z),
      &(state.ned_origin_f.hmsl), &(stateGetSpeedNed_f()->z),
      &(stateGetAccelNed_f()->z), &accel_ned_filt.o[0], &thrust,
      &autopilot.mode, &record);
}

// Function definitions
// Callback function of optical flow estimate (bound to optical flow ABI
// messages)
static void nl_optical_flow_cb(uint8_t sender_id, uint32_t stamp,
                               int16_t UNUSED flow_x, int16_t UNUSED flow_y,
                               int16_t UNUSED flow_der_x,
                               int16_t UNUSED flow_der_y, float UNUSED quality,
                               float size_divergence);

// NN landing module functions
static void nl_init(void);
static void nl_run(float divergence, float divergence_dot, float dt);

// Closed-loop, active thrust control
static void nl_control(void);

// Init global variables
static void init_globals(void);

// Network functions
// Zeroing neurons
static void zero_neurons(void){
  for (int16_t i = 0; i < nr_input_neurons; i++){
    input_layer_out[i] = 0.f;
  }
  for (int16_t i = 0; i < nr_hidden_neurons; i++){
    hidden_layer_out[i] = 0.f;
  }
  for (int16_t i = 0; i < nr_output_neurons; i++){
    layer2_out[i] = 0.f;
  }
}

// Run the various network types
static float predict_nn(float in[], float dt)
{
  int i,j;

  for (i = 0; i < nr_input_neurons; i++){
#if NN_TYPE == NN
    input_layer_out[i] = in[i] + bias0[i];
#elif NN_TYPE == RNN
    input_layer_out[i] = in[i] + bias0[i] + input_layer_out[i]*recurrent_weights0[i];
#elif NN_TYPE == CTRNN
    input_layer_out[i] += (in[i] - input_layer_out[i]) * dt / (dt + time_const0[i]);
#endif
  }

  float potential;
#if NN_TYPE == NN || NN_TYPE == RNN
  for (i = 0; i < nr_hidden_neurons; i++){
    potential = 0.f;
    for (j = 0; j < nr_input_neurons; j++){
      potential += input_layer_out[j]*layer1_weights[j][i];
    }
#if NN_TYPE == RNN
    potential += hidden_layer_out[i]*recurrent_weights1[i];
#endif
    hidden_layer_out[i] = relu(potential + bias1[i]);
  }

  for (i = 0; i < nr_output_neurons; i++){
    potential = 0.f;
    for (j = 0; j < nr_hidden_neurons; j++){
      potential += hidden_layer_out[j]*layer2_weights[j][i];
    }
#if NN_TYPE == RNN
    potential += layer2_out[i]*recurrent_weights2[i];
#endif
    layer2_out[i] = potential + bias2[i];
  }

#elif NN_TYPE == CTRNN
  for (i = 0; i < nr_hidden_neurons; i++) {
    potential = 0.f;
    for (j = 0; j < nr_input_neurons; j++) {
      potential += tanhf(gain0[j]*(input_layer_out[j] + bias0[j]))*layer1_weights[j][i];
    }
    hidden_layer_out[i] += (potential - hidden_layer_out[i]) * dt / (time_const1[i] + dt);
  }

  for (i = 0; i < nr_output_neurons; i++) {
    potential = 0.f;
    for (j = 0; j < nr_hidden_neurons; j++) {
      potential += tanhf(gain1[j]*(hidden_layer_out[j] + bias1[j]))*layer2_weights[j][i];
    }
    layer2_out[i] += (potential - layer2_out[i]) * dt / (time_const2[i] + dt);
  }
#endif

  return layer2_out[0];
}

// Module initialization function
static void nl_init() {

#ifndef NL_UART_CONTROL  
  // Init network
  zero_neurons();
#endif

  // Fill settings
  nl_settings.thrust_effect = NL_THRUST_EFFECT;
  nl_settings.thrust_p_gain = NL_THRUST_P_GAIN;
  nl_settings.thrust_i_gain = NL_THRUST_I_GAIN;

  // Init global variables
  init_globals();

  // Register telemetry message
  register_periodic_telemetry(DefaultPeriodic, PPRZ_MSG_ID_SPIKING_LANDING,
                              send_nl);

  // Subscribe to optical flow estimation
  AbiBindMsgOPTICAL_FLOW(NL_OPTICAL_FLOW_ID, &optical_flow_ev,
                         nl_optical_flow_cb);

  // Init low-pass filters for acceleration and thrust
  // float tau = 1.0f / (2.0f * M_PI * NL_OF_FILTER_CUTOFF);
  // In case of higher loop rate this seems too much, revert to above
  float tau = 5.0f;
  float ts = 1.0f / PERIODIC_FREQUENCY;
  init_butterworth_2_low_pass(&accel_ned_filt, tau, ts, 0.0f);
  init_butterworth_2_low_pass(&thrust_filt, tau, ts, 0.0f);
}

// Reset global variables (e.g., when starting/re-entering module)
static void init_globals() {
  divergence = 0.0f;
  divergence_half = 0.0f;
  divergence_dot = 0.0f;
  divergence_dot_half = 0.0f;
  div_gt = 0.0f;
  divdot_gt = 0.0f;
  div_gt_tmp = 0.0f;
  thrust = 0.0f;
  spike_count = 0;
  acc_lp = 0.0f;
  thrust_lp = 0.0f;
  acceleration_sp = 0.0f;
  record = 0;
  nominal_throttle = guidance_v_nominal_throttle;
  active_control = false;
}

// Get optical flow estimate from sensors via callback
static void nl_optical_flow_cb(uint8_t sender_id, uint32_t stamp,
                               int16_t UNUSED flow_x, int16_t UNUSED flow_y,
                               int16_t UNUSED flow_der_x,
                               int16_t UNUSED flow_der_y, float UNUSED quality,
                               float size_divergence) {
  // Compute time step
  static uint32_t last_stamp = 0;
  float dt = (stamp - last_stamp) / 1e6f;
  last_stamp = stamp;

  // Compute derivative of divergence and divergence
  if (dt > 1e-5f) {
    divergence_dot_half = (size_divergence - divergence_half) / dt;
    divergence_dot = (2.0f * size_divergence - divergence) / dt;
  }
  divergence_half = size_divergence;
  divergence = 2.0f * size_divergence;

  // Compute GT of divergence + derivative
  if (fabsf(stateGetPositionNed_f()->z) > 1e-5f) {
    div_gt_tmp = -2.0f * stateGetSpeedNed_f()->z / stateGetPositionNed_f()->z;
  }
  if (dt > 1e-5f) {
    divdot_gt = (div_gt_tmp - div_gt) / dt;
  }
  div_gt = div_gt_tmp;

  // Run the network
  nl_run(divergence_half, divergence_dot_half, dt);
}

// Run the network
static void nl_run(float divergence, float divergence_dot, float dt) {
  // These "static" types are great!
  static bool first_run = true;
  static float start_time = 0.0f;
  static float nominal_throttle_sum = 0.0f;
  static float nominal_throttle_samples = 0.0f;

  // TODO: is this for resetting altitude?
  if (autopilot_get_mode() != AP_MODE_GUIDED) {
    first_run = true;
    active_control = false;
    record = 0;
    return;
  }

  // TODO: here we reset the network in between runs!
  if (first_run) {
    start_time = get_sys_time_float();
    nominal_throttle = (float)stabilization_cmd[COMMAND_THRUST] / MAX_PPRZ;
#ifdef NL_UART_CONTROL
    uart_driver_tx_event(divergence, (uint8_t)1);
#else
    zero_neurons();
#endif
    first_run = false;
  }

  // Let the vehicle settle
  if (get_sys_time_float() - start_time < 5.0f) {
    return;
  }

  // After vehicle settling, compute and improve nominal throttle estimate
  if (get_sys_time_float() - start_time < 10.0f) {
    nominal_throttle_sum += (float)stabilization_cmd[COMMAND_THRUST] / MAX_PPRZ;
    nominal_throttle_samples++;
    nominal_throttle = nominal_throttle_sum / nominal_throttle_samples;

    // Initialize network by running zeros through it
    static float zero_input[] = {0.f, 0.f};
    predict_nn(zero_input, dt);
    return;
  }

  // Set recording while in flight
  if (autopilot.in_flight) {
    record = 1;
  } else {
    record = 0;
  }

  // SNN onboard paparazzi
  // ifdef:
  // - Send divergence to upboard over UART
  // - UART event-triggered RX loop overwrites thrust (see "modules/uart_driver/uart_driver.c")
#ifdef NL_UART_CONTROL
  uart_driver_tx_event(divergence, (uint8_t)0);
#else
  // Forward network to get action/thrust for control
  float input[] = {divergence, divergence_dot};
  thrust = predict_nn(input, dt);

  // Bound thrust to limits (-0.8g, 0.5g)
  Bound(thrust, -7.848f, 4.905f);
#endif

  // Set control mode: active closed-loop control or linear transform
  if (NL_ACTIVE_CONTROL) {
    active_control = true;
  } else {
    guidance_v_set_guided_th(thrust * nl_settings.thrust_effect +
                             nominal_throttle);
  }
}

// Closed-loop PI control for going from acceleration to motor control
static void nl_control() {
  // "static" here implies that value is kept between function invocations
  static float error_integrator = 0.0f;

  // Low-pass filters for current acceleration and thrust setpoint
  struct NedCoor_f *acceleration = stateGetAccelNed_f();
  update_butterworth_2_low_pass(&accel_ned_filt, acceleration->z);
  update_butterworth_2_low_pass(&thrust_filt, thrust);
  acc_lp = accel_ned_filt.o[0];
  thrust_lp = thrust_filt.o[0];

  // Proportional
  /**
   * Acceleration is used in a meaningful way I think?
   * TODO: + because of negative accel for up?
   * TODO: why this bound? --> in Kirk's code it says to limit effect of integrator to 1m/s
   * TODO: why thrust effectiveness?
   */
  float error = thrust_filt.o[0] + accel_ned_filt.o[0];
  BoundAbs(error, 1.0f / (nl_settings.thrust_p_gain + 0.01f));

  // Integral
  error_integrator += error / PERIODIC_FREQUENCY;
  BoundAbs(error_integrator, 1.0f / (nl_settings.thrust_i_gain + 0.01f));

  // Acceleration setpoint
  acceleration_sp = (thrust + error * nl_settings.thrust_p_gain +
                           error_integrator * nl_settings.thrust_i_gain) *
                              nl_settings.thrust_effect +
                          nominal_throttle;

  // Perform active closed-loop control or do simple linear transform
  if (active_control) {
    guidance_v_set_guided_th(acceleration_sp);
  } else {
    error_integrator = 0.0f;
  }
}

// Module functions
// Init
void nn_landing_init() { nl_init(); }

// Run
void nn_landing_event() { nl_control(); }
