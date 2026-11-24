; X/Y replace v6. A lane-dependent high half in v7 cannot reach an export of v6.
s_mov_b64 s[10:11], exec
v_mov_b32_e32 v6, 0
v_mbcnt_lo_u32_b32 v7, -1, 0
v_mbcnt_hi_u32_b32 v7, -1, v7
v_lshlrev_b32_e32 v7, 16, v7
s_mov_b32 s4, 0
s_mov_b32 s5, 0
s_mov_b32 s6, 0x10000
s_mov_b32 s7, 0x31045fac
v_mov_b32_e32 v5, 0x100
buffer_load_format_d16_xyz v[6:7], v5, s[4:7], 0 offen
s_waitcnt vmcnt(0)
exp mrt0 v6, v6, v6, v6 done vm
s_endpgm
