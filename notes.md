# Performance metrics
Saturation rate of math throughput to code
- SR := saturation_rate ;
- SR = 76%

Gelems correctly scaled what criterion shows
- gflops = 50.0; 
How much of your 4-core chip you are using (Single Core = 25%)
-  core_utilization = 25%; 

Peak math capability of one core at base speed  2.2 GHZ * 32
- max_unboosted_per_core = 70.4;

Peak math capability of one core at max turbo  3.4 GHZ * 32
-  max_boosted_per_core = 108.8;

<hat> SR[unboosted] = gflops * core_utilization / maxGhz[unboosted];
<hat> SR[unboosted] = gflops * core_utilization / maxGhz[boosted];

<hat> GHZ = gflops  * core_utilization / SR

# Common commands

```
RUSTFLAGS="-C target-cpu=native" cargo rustc --lib --features="avx2" --release -- --emit=asm
```
```
find ./target/release/deps -name \"*.s\"
```

then view the output

# ASM instructions
vfmadd213ph	// ph -> packed half
vfmadd213ps	// ps -> packed single
vfmadd213pd	// pd -> packed double

physically casted instructions are called
hard-wired execution unit; // ie like instructions that are gate networks
