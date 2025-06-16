fn nd_offsets(dims:&[usize]) -> Vec<usize> {
    let mut init = 1;
    let mut factors = Vec::with_capacity(dims.len());
    for &num in dims[0..].iter().rev() {
        factors.push(init);
        init *= num;
    }
    factors
}

fn transpose<T>(data:&mut [T]) {
    let dims = vec![2,2,2];
    let card = dims.len();
    let offsets = nd_offsets(&dims);
    // running product of dimensions
    let mut mem = vec![false;dims.iter().product()];
    let mut state = vec![0;card];
    let mut cursor = 0;
    
    // iterates over the dimensions
    while cursor < card {
        let mut targets = vec![0;card];
        for idx in 0..card {
            for (cdx, s) in state.iter().enumerate() {
                // let a = k*(x*y) + i*y + j;
                // let b = j*(x*y) + k*y + i;
                // let c = i*(x*y) + j*y + k;
                targets[idx]+=s*offsets[(idx + cdx) % card];
            }
        }
        if !mem[targets[0]] {
            mem[targets[0]] = true;
            for idx in 1..card {
                mem[targets[idx]] = true;
                data.swap(targets[idx-1], targets[idx]);
            }
        }
        loop {
            if cursor == card {
                break;
            }
            if state[cursor] + 1 < dims[cursor] {
                state[cursor] += 1;
                for idx in 0..cursor {
                    state[idx] = 0;
                }
                cursor=0;
                break;
            } else {
                cursor += 1;
            }
        }
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
    transpose(data);
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
