# Machine Specs Validation Report

Generated: 2026-08-04T13:44:55+02:00
Host: legion

## System

### hostnamectl
```text
  Static hostname: legion
        Icon name: computer-laptop
          Chassis: laptop 💻
Chassis Asset Tag: NO Asset Tag
       Machine ID: 91f7f5aa839a40fc85a55be353d79f59
          Boot ID: e0e0583bb87f4cee950eea8609dcfdae
 Operating System: Arch Linux
           Kernel: Linux 7.1.5-arch1-2
     Architecture: x86-64
  Hardware Vendor: Lenovo
   Hardware Model: Legion 5 15AHP10
     Hardware SKU: LENOVO_MT_83M0_BU_idea_FM_Legion 5 15AHP10
 Firmware Version: RGCN34WW
    Firmware Date: Thu 2025-12-11
     Firmware Age: 7month 3w 2d
```

### Kernel
```text
7.1.5-arch1-2
```

### OS Release
```text
NAME="Arch Linux"
PRETTY_NAME="Arch Linux"
ID=arch
BUILD_ID=rolling
ANSI_COLOR="38;2;23;147;209"
HOME_URL="https://archlinux.org/"
DOCUMENTATION_URL="https://wiki.archlinux.org/"
SUPPORT_URL="https://bbs.archlinux.org/"
BUG_REPORT_URL="https://gitlab.archlinux.org/groups/archlinux/-/issues"
PRIVACY_POLICY_URL="https://terms.archlinux.org/docs/privacy-policy/"
LOGO=archlinux-logo
```

### Desktop Session
```text
OutputOrderWatcher may not work as expected. Reason: kde_output_order_v1 protocol is not available
plasmashell 6.7.3
Failed to get path for session '': Caller does not belong to any known session.
[command failed] bash -lc loginctl show-session "$XDG_SESSION_ID" -p Type -p Desktop -p Name
```

## Memory

### free -h
```text
               total        used        free      shared  buff/cache   available
Mem:            30Gi       6.0Gi        21Gi        55Mi       4.1Gi        24Gi
Swap:           15Gi          0B        15Gi
```

### Memory Channel Topology (dmidecode)
```text
# dmidecode 3.7
Getting SMBIOS data from sysfs.
SMBIOS 3.6.0 present.

Handle 0x001F, DMI type 16, 23 bytes
Physical Memory Array
	Location: System Board Or Motherboard
	Use: System Memory
	Error Correction Type: None
	Maximum Capacity: 48 GiB
	Error Information Handle: 0x0022
	Number Of Devices: 2

Handle 0x0020, DMI type 17, 92 bytes
Memory Device
	Array Handle: 0x001F
	Error Information Handle: 0x0023
	Total Width: 64 bits
	Data Width: 64 bits
	Size: 16 GiB
	Form Factor: SODIMM
	Set: None
	Locator: DIMM 0
	Bank Locator: P0 CHANNEL A
	Type: DDR5
	Type Detail: Synchronous Unbuffered (Unregistered)
	Speed: 5600 MT/s
	Manufacturer: Micron Technology
	Serial Number: 502C943C
	Asset Tag: Not Specified
	Part Number: MTC8C1084S1SC56BD1 K
	Rank: 1
	Configured Memory Speed: 5600 MT/s
	Minimum Voltage: 1.1 V
	Maximum Voltage: 1.1 V
	Configured Voltage: 1.1 V
	Memory Technology: DRAM
	Memory Operating Mode Capability: Volatile memory
	Firmware Version: Unknown
	Module Manufacturer ID: Bank 1, Hex 0x2C
	Module Product ID: Unknown
	Memory Subsystem Controller Manufacturer ID: Unknown
	Memory Subsystem Controller Product ID: Unknown
	Non-Volatile Size: None
	Volatile Size: 16 GiB
	Cache Size: None
	Logical Size: None

Handle 0x0021, DMI type 17, 92 bytes
Memory Device
	Array Handle: 0x001F
	Error Information Handle: 0x0024
	Total Width: 64 bits
	Data Width: 64 bits
	Size: 16 GiB
	Form Factor: SODIMM
	Set: None
	Locator: DIMM 0
	Bank Locator: P0 CHANNEL B
	Type: DDR5
	Type Detail: Synchronous Unbuffered (Unregistered)
	Speed: 5600 MT/s
	Manufacturer: Unknown
	Serial Number: E9FD1C6A
	Asset Tag: Not Specified
	Part Number: CT16G56C46S5.C8D    
	Rank: 1
	Configured Memory Speed: 5600 MT/s
	Minimum Voltage: 1.1 V
	Maximum Voltage: 1.1 V
	Configured Voltage: 1.1 V
	Memory Technology: DRAM
	Memory Operating Mode Capability: Volatile memory
	Firmware Version: Unknown
	Module Manufacturer ID: Bank 6, Hex 0x9B
	Module Product ID: Unknown
	Memory Subsystem Controller Manufacturer ID: Unknown
	Memory Subsystem Controller Product ID: Unknown
	Non-Volatile Size: None
	Volatile Size: 16 GiB
	Cache Size: None
	Logical Size: None

```

## Storage

### lsblk
```text
NAME          SIZE TYPE MOUNTPOINT FSTYPE MODEL
nvme0n1     476.9G disk                   WD PC SN7100S SDFPMSL-512G-1101
├─nvme0n1p1   100M part /boot/efi  vfat   
├─nvme0n1p2    16G part [SWAP]     swap   
└─nvme0n1p3 460.8G part /          ext4   
nvme1n1     931.5G disk                   Lexar SSD NM710 1TB
├─nvme1n1p1   260M part            vfat   
├─nvme1n1p2    16M part                   
├─nvme1n1p3 929.2G part            ntfs   
└─nvme1n1p4     2G part            ntfs   
```

## CPU

### lscpu
```text
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           48 bits physical, 48 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  16
On-line CPU(s) list:                     0-15
Vendor ID:                               AuthenticAMD
Model name:                              AMD Ryzen 7 260 w/ Radeon 780M Graphics
CPU family:                              25
Model:                                   117
Thread(s) per core:                      2
Core(s) per socket:                      8
Socket(s):                               1
Stepping:                                2
Microcode version:                       0xa70520a
Frequency boost:                         enabled
CPU(s) scaling MHz:                      38%
CPU max MHz:                             5102.7129
CPU min MHz:                             416.5480
BogoMIPS:                                7585.63
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl xtopology nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpuid_fault cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid overflow_recov succor smca fsrm flush_l1d amd_lbr_pmc_freeze
Virtualization:                          AMD-V
L1d cache:                               256 KiB (8 instances)
L1i cache:                               256 KiB (8 instances)
L2 cache:                                8 MiB (8 instances)
L3 cache:                                16 MiB (1 instance)
NUMA node(s):                            1
NUMA node0 CPU(s):                       0-15
Vulnerability Gather data sampling:      Not affected
Vulnerability Ghostwrite:                Not affected
Vulnerability Indirect target selection: Not affected
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Not affected
Vulnerability Mds:                       Not affected
Vulnerability Meltdown:                  Not affected
Vulnerability Mmio stale data:           Not affected
Vulnerability Old microcode:             Not affected
Vulnerability Reg file data sampling:    Not affected
Vulnerability Retbleed:                  Not affected
Vulnerability Spec rstack overflow:      Mitigation; Safe RET
Vulnerability Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:                Mitigation; Enhanced / Automatic IBRS; IBPB conditional; STIBP always-on; PBRSB-eIBRS Not affected; BHI Not affected
Vulnerability Srbds:                     Not affected
Vulnerability Tsa:                       Mitigation; Clear CPU buffers
Vulnerability Tsx async abort:           Not affected
Vulnerability Vmscape:                   Mitigation; IBPB before exit to userspace
```

### Powercap
```text
total 0
drwxr-xr-x  2 root root 0 Aug  4 13:44 .
drwxr-xr-x 80 root root 0 Aug  4 09:40 ..
lrwxrwxrwx  1 root root 0 Aug  4 13:41 intel-rapl -> ../../devices/virtual/powercap/intel-rapl
lrwxrwxrwx  1 root root 0 Aug  4 13:41 intel-rapl:0 -> ../../devices/virtual/powercap/intel-rapl/intel-rapl:0
lrwxrwxrwx  1 root root 0 Aug  4 13:41 intel-rapl:0:0 -> ../../devices/virtual/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0

-- energy paths --
/sys/class/powercap/intel-rapl:0:0/energy_uj
/sys/class/powercap/intel-rapl:0:0/device/energy_uj
/sys/class/powercap/intel-rapl:0:0/device/max_energy_range_uj
/sys/class/powercap/intel-rapl:0:0/max_energy_range_uj
/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj
/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/max_energy_range_uj
/sys/class/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj
/sys/class/powercap/intel-rapl:0/energy_uj
/sys/class/powercap/intel-rapl:0/intel-rapl:0:0/energy_uj
/sys/class/powercap/intel-rapl:0/intel-rapl:0:0/max_energy_range_uj
/sys/class/powercap/intel-rapl:0/max_energy_range_uj
```

## GPU

### nvidia-smi query
```text
NVIDIA GeForce RTX 5060 Laptop GPU, 610.43.03, 8151 MiB, 50.00 W, 115.00 W
```

### nvidia-smi summary
```text
Tue Aug  4 13:44:55 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.43.03              KMD Version: 610.43.03     CUDA UMD Version: 13.3     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5060 ...    Off |   00000000:01:00.0 Off |                  N/A |
| N/A   38C    P8              8W /   50W |     235MiB /   8151MiB |      1%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            3219      G   /usr/bin/gnome-shell                      2MiB |
|    0   N/A  N/A            4187    C+G   /usr/lib/ibus/ibus-x11                    5MiB |
```

## NPU

### Driver and device nodes
```text
amdxdna               225280  0

/dev/accel
total 0
drwxr-xr-x  2 root root       60 Aug  4 09:40 .
drwxr-xr-x 22 root root     4880 Aug  4 11:31 ..
crw-rw-rw-  1 root render 261, 0 Aug  4 09:40 accel0

/dev/dri
total 0
drwxr-xr-x   3 root root        140 Aug  4 09:40 .
drwxr-xr-x  22 root root       4880 Aug  4 11:31 ..
drwxr-xr-x   2 root root        120 Aug  4 09:40 by-path
crw-rw----+  1 root video  226,   1 Aug  4 09:41 card1
crw-rw----+  1 root video  226,   2 Aug  4 09:41 card2
crw-rw-rw-   1 root render 226, 128 Aug  4 09:40 renderD128
crw-rw-rw-   1 root render 226, 129 Aug  4 09:40 renderD129
```

### xrt-smi examine
```text
System Configuration
  OS Name              : Linux
  Release              : 7.1.5-arch1-2
  Machine              : x86_64
  CPU Cores            : 16
  Memory               : 31384 MB
  Distribution         : Arch Linux
  GLIBC                : 2.44
  Model                : 83M0
  BIOS Vendor          : LENOVO
  BIOS Version         : RGCN34WW
  Processor            : AMD Ryzen 7 260 w/ Radeon 780M Graphics

XRT
  Version              : 2.21.75
  Branch               : makepkg
  Hash                 : 4eb1f4392a012b4e6eca759762389c612537f7c7
  Hash Date            : 2026-04-24 10:52:40
  virtio-pci Version   : 7.1.5-arch1-2
  amdxdna Version      : 7.1.5-arch1-2
  NPU Firmware Version : 1.5.5.391

Device(s) Present
|BDF             |Name          |
|----------------|--------------|
|[0000:66:00.1]  |RyzenAI-npu1  |


```

### NPU memory model evidence
```text
xrt-smi memory-related lines:
  Memory               : 31384 MB

lspci BAR regions for NPU (0000:66:00.1):
	Region 0: Memory at bc100000 (32-bit, non-prefetchable) [size=512K]
	Region 1: Memory at bc1c0000 (32-bit, non-prefetchable) [size=8K]
	Region 2: Memory at 7e02000000 (64-bit, prefetchable) [size=256K]
	Region 4: Memory at bc180000 (32-bit, non-prefetchable) [size=256K]
```

## Quick Validation Checklist

- CPU info present: yes
- GPU query available: yes
- NPU runtime query available: yes
- energy_uj path present: yes
- Installed RAM (MemTotal): 30.6 GiB
- Memory channel configuration (dmidecode): dual-channel (2 modules across 2 channels)
- NPU shared-system-memory indication: likely yes
