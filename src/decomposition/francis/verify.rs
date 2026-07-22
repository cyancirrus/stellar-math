use crate::algebra::ndmethods::matrix_mult;
use crate::decomposition::francis::primitives::{
    complex_eig_pair, deflate, double_shift, exception_shift,
};
// pub fn full_decomp_cpx(
//     h: &mut [f32],
//     _r: &mut [f32],
//     w: &mut [f32],
//     mut range: usize,
//     size: usize,
//     stride: usize,
// ) {
//     let s = range * stride;
//     let mut e1 = s.saturating_sub(stride + 1);
//     let mut e2 = s.saturating_sub(stride + stride + 2);
//     let mut tl = s.saturating_sub(stride + 2);
//     let mut bl = s.saturating_sub(2);
//     let mut curriter = 0;
//     let _he1 = h[e1];
//     let _he2 = h[e2];
//     let p = &mut [0f32; 3];
//     let mut stall = 0;
//     while range > 0 && curriter < MAX_ITERS {
//         curriter += 1;
//         if h[e1].abs() < TOLERANCE {
//             stall = 0;
//             deflate(
//                 1,
//                 stride,
//                 &mut range,
//                 &mut e1,
//                 &mut e2,
//                 &mut tl,
//                 &mut bl,
//                 &mut curriter,
//             );
//         } else if h[e2].abs() < TOLERANCE {
//             // if e2 == 0 then we are hitting eigen which should be greater than tolerance
//             deflate(
//                 2,
//                 stride,
//                 &mut range,
//                 &mut e1,
//                 &mut e2,
//                 &mut tl,
//                 &mut bl,
//                 &mut curriter,
//             );
//             stall = 0;
//         } else if range == 2 && complex_eig_pair(h, tl, bl) {
//             deflate(
//                 2,
//                 stride,
//                 &mut range,
//                 &mut e1,
//                 &mut e2,
//                 &mut tl,
//                 &mut bl,
//                 &mut curriter,
//             );
//             stall = 0;
//         } else {
//             if range == 2 {
//                 francis_iteration_cpx_2x2(h, size, stride, tl, bl);
//             // } else if stall > 0 && (stall + 4) % 10 == 0 {
//             } else if (stall + 8) % 12 == 0 {
//                 // } else if (stall + 4) % 10 == 0 {
//                 // } else if stall == 6 {
//                 exception_shift(h, w, stride, range, tl, bl);
//                 francis_iteration_cpx(h, p, w, size, range, stride, tl, bl);
//             } else {
//                 double_shift(h, w, stride, range, tl, bl);
//                 francis_iteration_cpx(h, p, w, size, range, stride, tl, bl);
//             }
//             stall += 1;
//         }
//     }
//     if range > 1 {
//         println!("missed");
//     }
// }
