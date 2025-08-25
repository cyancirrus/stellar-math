fn nd_offsets(dims:&[usize]) -> Vec<usize> {
    let mut init = 1;
    let mut factors = Vec::with_capacity(dims.len());
    for &num in dims[0..].iter().rev() {
        factors.push(init);
        init *= num;
    }
    factors
}

fn offset(dims:&[usize]) -> Vec<usize> {
    let mut init = 1;
    let n = dims.len();
    let mut idxs = vec![1;n];
    //[1,2,3,4]
    for id in (0..n-1).rev() {
        init *= dims[id];
        idxs[id] = init;
    }
    idxs
}


fn transpose_clean<T>(data:&mut[T]) {
    let dims = vec![2,2,2];
    //  coming in to the function
    let n = dims.iter().product();
    let card = dims.len();
    let offset = vec![4,2,1];
    let mut cycle = vec![0; card];
    let mut mem = vec![false; n];
    let mut state = vec![1;card];
    let mut cursor = 0;

    for _ in 0..n {
        if !mem[cycle[0]] {
            mem[cycle[0]] = true;
            for p in 1..card {
                mem[cycle[p]] = true;
                data.swap(cycle[p-1], cycle[p]);
            }
        }
        if state[cursor] < dims[cursor] {
            state[cursor] += 1;
            for g in 0..card {
                cycle[g] += offset[(cursor + g) % card];
            }
            continue;
        }
        loop {
            cursor += 1;
            if cursor == card {
                return;
            }
            if state[cursor] < dims[cursor] {
                // found an open slot
                state[cursor] += 1;
                // decrementing like in the 3d case
                for g in 0..card {
                    cycle[g] += offset[(cursor + g) % card];
                }
                break;
            }
        };
        for idx in 0..cursor {
            state[idx]=1;
            for g in 0..card {
                cycle[g] -= (dims[idx]-1) * offset[(idx + g) % card];
            }
        }
        cursor=0;
    }
}

fn transpose_ideation<T>(data:&mut[T]) {
    let dims = vec![2,2,2];
    let offset = [4,2,1];
    let card = dims.len();
    let mut cycle = [0,0,0];
    let mut mem = vec![false; dims.iter().product()];
    let mut s;
    for _ in 0..dims[0] {
        for _ in 0..dims[1] {
            for _ in 0..dims[2] {
                if !mem[cycle[0]] {
                    mem[cycle[0]]=true;
                    for i in 1..card {
                        data.swap(cycle[i-1], cycle[i]);
                        mem[cycle[i]]=true;
                    }
                }
                // variable 2 + offset 2
                cycle[0] += offset[2];
                cycle[1] += offset[0];
                cycle[2] += offset[1];
            }
            // variable 2 for - offset 2
            s = dims[2];
            cycle[0] -= s * offset[2];
            cycle[1] -= s * offset[0];
            cycle[2] -= s * offset[1];
            // variable 1 for + offset 1
            cycle[0] += offset[1];
            cycle[1] += offset[2];
            cycle[2] += offset[0];
        }
        // variable 1 for - offset 1
        s = dims[1];
        cycle[0] -= s * offset[1];
        cycle[1] -= s * offset[2];
        cycle[2] -= s * offset[0];
        // variable 0 for + offset 0
        cycle[0] += offset[0];
        cycle[1] += offset[1];
        cycle[2] += offset[2];
    }
}



fn pretty_print(data:&[usize]) {
    println!("----------");
    for i in (0..8).step_by(2) {
        if i == 4 {
            println!(" - - - - - -");
        }
        println!("[ {:03b}, {:03b} ]", data[i], data[i+1]);
    }
    println!("----------");
}

fn debug_transpose(data:&mut [usize]) {
    transpose_clean(data);
    pretty_print(data);
}

fn debug_data() -> Vec<usize> {
    // dimensions indexed by (x,y,z)
    vec![
        0, 2,
        4, 6,
        //---
        1, 3,
        5, 7 
    ]
}

// fn debug_data4d() -> Vec<usize> {
//     // t, z, x, y
//     vec![
//         0b0000, 0b0001,
//         0b0010, 0b0011,
//         //
//         0b0100, 0b0101,
//         0b0110, 0b0111,
//         //
//         0b1010, 

//     ]

// }

fn main() {
    let data = debug_data();
    let mut test = debug_data();
    pretty_print(&test);
    debug_transpose(&mut test);
    debug_transpose(&mut test);
    debug_transpose(&mut test);
    assert_eq!(data, test);

    // thingy();

    // // println!("dims {:?}", get_inner(&[5,4,3,2,1]));
    // assert_eq!(vec![1, 2, 6, 24], get_inner(&[5,4,3,2,1]));
    // let mut data = [1,2,3,4];
    // transpose_test(&mut data);
    // assert_eq!(data, [2,3,4,1]);
    // transpose_test(&mut data);
    // assert_eq!(data, [3,4,1,2]);
    // transpose_test(&mut data);
    // assert_eq!(data, [4,1,2,3]);
    // transpose_test(&mut data);
    // assert_eq!(data, [1,2,3,4]);
    // println!("new prototype function?");
}

// fn transpose_ideation<T>(data:&mut[T]) {
//     let dims = vec![2,2,2];
//     let offset = [4,2,1];
//     let card = dims.len();
//     let mut incrementer = [0,0,0];
//     let mut mem = vec![false; dims.iter().product()];
//     let mut s;
//     for i in 0..dims[0] {
//         if i > 0 {
//             incrementer[0] += offset[0];
//             incrementer[1] += offset[1];
//             incrementer[2] += offset[2];
//         }
//         for j in 0..dims[1] {
//             if j > 0 {
//                 incrementer[0] += offset[1];
//                 incrementer[1] += offset[2];
//                 incrementer[2] += offset[0];
//             }
//             for k in 0..dims[2] {
//                 if k > 0 {
//                     incrementer[0] += offset[2];
//                     incrementer[1] += offset[0];
//                     incrementer[2] += offset[1];
//                 }
//                 if !mem[incrementer[0]] {
//                     mem[incrementer[0]]=true;
//                     for i in 1..card {
//                         data.swap(incrementer[i-1], incrementer[i]);
//                         mem[incrementer[i]]=true;
//                     }
//                 }
//             }
//             s = dims[2] - 1;
//             incrementer[0] -= s * offset[2];
//             incrementer[1] -= s * offset[0];
//             incrementer[2] -= s * offset[1];
//         }
//         s = dims[1] - 1;
//         incrementer[0] -= s * offset[1];
//         incrementer[1] -= s * offset[2];
//         incrementer[2] -= s * offset[0];
//     }
// }


// fn transpose<T>(data:&mut [T]) {
//     let dims = vec![2,2,2];
//     let card = dims.len();
//     let offsets = nd_offsets(&dims);
//     // running product of dimensions
//     let mut mem = vec![false; dims.iter().product()];
//     let mut state = vec![0;card];
//     let mut cursor = 0;
//     let mut targets = vec![0;card];
    
//     // iterates over the dimensions
//     while cursor < card {
//         targets.fill(0);
//         for idx in 0..card {
//             for (cdx, s) in state.iter().enumerate() {
//                 // let a = k*(x*y) + i*y + j;
//                 // let b = j*(x*y) + k*y + i;
//                 // let c = i*(x*y) + j*y + k;
//                 targets[idx]+=s*offsets[(idx + cdx) % card];
//             }
//         }
//         if !mem[targets[0]] {
//             mem[targets[0]] = true;
//             for idx in 1..card {
//                 mem[targets[idx]] = true;
//                 data.swap(targets[idx-1], targets[idx]);
//             }
//         }
//         loop {
//             if cursor == card {
//                 break;
//             }
//             if state[cursor] + 1 < dims[cursor] {
//                 state[cursor] += 1;
//                 for idx in 0..cursor {
//                     state[idx] = 0;
//                 }
//                 cursor=0;
//                 break;
//             } else {
//                 cursor += 1;
//             }
//         }
//     }
// }
