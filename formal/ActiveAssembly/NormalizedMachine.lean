/-!
# Normalized bounded machine state

This file is the Lean-side schema for the promotion tranche's explicit
machine state.  It intentionally models a bounded register/memory interface;
it is not a claim that the whole x86-64 ISA has been formalized.  The Python
reference interpreter and the Unicorn receipt use the corresponding fields
and the same width/flag boundary.

The state transformers below mirror the declared Python semantics in
``src/formal_correspondence.py`` exactly: the flag helpers mirror
``_set_arithmetic_flags``/``_set_logic_flags`` (including AF being marked
undefined by logic operations), the width moves mirror the register-write
width rules, the stack transformers mirror the bounded push/pop window and
little-endian byte order, and the conditional-branch transformer mirrors the
fail-closed undefined-flag read.
-/

namespace ActiveAssembly

def MOD64N : Nat := 2 ^ 64

theorem mod64n_pos : 0 < MOD64N := by
  simp [MOD64N]

def Word64N := Fin MOD64N

def word64n (value : Nat) : Word64N :=
  ⟨value % MOD64N, Nat.mod_lt _ mod64n_pos⟩

inductive MachineStatusN where
  | running
  | halted
  | unsupported
  | boundExceeded
  | exceptional
deriving Repr, DecidableEq

structure FlagsN where
  cf : Bool
  pf : Bool
  af : Bool
  zf : Bool
  sf : Bool
  of : Bool
  /-- Definedness keyed by the architectural eflags bit position, mirroring the
  Python ``flags_defined`` map (``FLAG_BITS`` in ``src/formal_correspondence.py``). -/
  defined : Nat → Bool


structure MachineStateN where
  gprs : Fin 16 → Word64N
  rip : Word64N
  flags : FlagsN
  memory : Nat → UInt8
  stackLower : Nat
  stackUpper : Nat
  memoryLower : Nat
  memoryUpper : Nat
  stepIndex : Nat
  status : MachineStatusN


def write32N (_old : Word64N) (value : Nat) : Word64N :=
  word64n (value % (2 ^ 32))

theorem write32_zero_extends (old : Word64N) (value : Nat) :
    (write32N old value).val = value % (2 ^ 32) := by
  change (value % (2 ^ 32)) % MOD64N = value % (2 ^ 32)
  apply Nat.mod_eq_of_lt
  exact Nat.lt_of_lt_of_le (Nat.mod_lt _ (by omega)) (by simp [MOD64N])

def inc64N (value : Word64N) : Word64N :=
  word64n (value.val + 1)

def incStateN (state : MachineStateN) : MachineStateN :=
  { state with
    gprs := fun index => if index = 0 then inc64N (state.gprs index) else state.gprs index
    stepIndex := state.stepIndex + 1 }

theorem inc_preserves_cf (state : MachineStateN) :
    (incStateN state).flags.cf = state.flags.cf := by
  rfl

def adc64N (left right : Word64N) (carry : Bool) : Word64N :=
  word64n (left.val + right.val + if carry then 1 else 0)

theorem adc_zero_carry (left right : Word64N) :
    adc64N left right false = word64n (left.val + right.val) := by
  simp [adc64N]

def memoryInBoundsN (state : MachineStateN) (address width : Nat) : Prop :=
  state.memoryLower ≤ address ∧ address + width ≤ state.memoryUpper

theorem unsupported_is_explicit :
    MachineStatusN.unsupported ≠ MachineStatusN.running := by
  decide

/-! ## Modeled eflags bit positions and register indices

The bit positions mirror ``FLAG_BITS`` and the register indices mirror
``GENERAL_REGISTER_NAMES`` in ``src/formal_correspondence.py`` exactly. -/

def cfBitN : Nat := 0
def pfBitN : Nat := 2
def afBitN : Nat := 4
def zfBitN : Nat := 6
def sfBitN : Nat := 7
def ofBitN : Nat := 11

def raxIndexN : Fin 16 := ⟨0, by omega⟩
def rbxIndexN : Fin 16 := ⟨1, by omega⟩
def rbpIndexN : Fin 16 := ⟨6, by omega⟩
def rspIndexN : Fin 16 := ⟨7, by omega⟩

/-- Parity of the low byte: even number of set bits, exactly like
``_parity_even`` (``(value & 0xFF).bit_count() % 2 == 0``).  The xor-fold
computes the parity of all eight low bits. -/
def parityEvenN (value : Nat) : Bool :=
  let byte := value % 256
  let first := Nat.xor byte (byte / 2)
  let second := Nat.xor first (first / 4)
  let third := Nat.xor second (second / 16)
  third % 2 == 0

/-- Flag set after a 64-bit modular addition, mirroring
``_set_arithmetic_flags`` with ``subtract=False`` and zero carry-in: every
modeled flag is defined. -/
def addFlagsN (left right : Word64N) : FlagsN :=
  let result := word64n (left.val + right.val)
  { cf := left.val + right.val ≥ MOD64N
    pf := parityEvenN result.val
    af := left.val % 16 + right.val % 16 > 15
    zf := result.val = 0
    sf := 2 ^ 63 ≤ result.val
    of := ((2 ^ 63 ≤ left.val) == (2 ^ 63 ≤ right.val)) &&
          ((2 ^ 63 ≤ result.val) != (2 ^ 63 ≤ left.val))
    defined := fun _ => true }

/-- Flag set after a 64-bit modular subtraction, mirroring
``_set_arithmetic_flags`` with ``subtract=True`` and zero carry-in: every
modeled flag is defined. -/
def subFlagsN (left right : Word64N) : FlagsN :=
  let result := word64n (left.val + MOD64N - right.val)
  { cf := left.val < right.val
    pf := parityEvenN result.val
    af := left.val % 16 < right.val % 16
    zf := result.val = 0
    sf := 2 ^ 63 ≤ result.val
    of := ((2 ^ 63 ≤ left.val) != (2 ^ 63 ≤ right.val)) &&
          ((2 ^ 63 ≤ result.val) != (2 ^ 63 ≤ left.val))
    defined := fun _ => true }

/-- Flag set after a logic operation, mirroring ``_set_logic_flags``: CF and OF
are defined and cleared, ZF/SF/PF are defined from the result, and AF carries
the value ``false`` while being marked UNDEFINED. -/
def logicFlagsN (result : Word64N) : FlagsN :=
  { cf := false
    pf := parityEvenN result.val
    af := false
    zf := result.val = 0
    sf := 2 ^ 63 ≤ result.val
    of := false
    defined := fun bit =>
      (bit == cfBitN) || (bit == pfBitN) || (bit == zfBitN) || (bit == sfBitN)
        || (bit == ofBitN) }

theorem arithmetic_flags_fully_defined (left right : Word64N) (bit : Nat) :
    (addFlagsN left right).defined bit = true ∧
    (subFlagsN left right).defined bit = true := by
  simp [addFlagsN, subFlagsN]

theorem logic_flags_af_undefined (result : Word64N) :
    (logicFlagsN result).defined afBitN = false := by
  show ((afBitN == cfBitN) || (afBitN == pfBitN) || (afBitN == zfBitN)
    || (afBitN == sfBitN) || (afBitN == ofBitN)) = false
  decide

theorem logic_flags_cf_defined_and_clear (result : Word64N) :
    (logicFlagsN result).defined cfBitN = true ∧ (logicFlagsN result).cf = false := by
  refine ⟨?_, rfl⟩
  show ((cfBitN == cfBitN) || (cfBitN == pfBitN) || (cfBitN == zfBitN)
    || (cfBitN == sfBitN) || (cfBitN == ofBitN)) = true
  decide

/-! ## Width-aware move semantics

These mirror the Python width rules for register writes: 8/16-bit writes
preserve the upper bits of the destination, a 32-bit write zero-extends into
the full 64-bit register (``write32N`` above), and a 64-bit write replaces the
whole register.  ``movzxN``/``movsxN`` mirror the zero/sign extension rules. -/

def mov8N (old : Word64N) (value : Nat) : Word64N :=
  word64n ((old.val / 256) * 256 + value % 256)

def mov16N (old : Word64N) (value : Nat) : Word64N :=
  word64n ((old.val / 65536) * 65536 + value % 65536)

/-- ``movzx`` zero-extension from ``sourceWidth`` bits into the 64-bit word. -/
def movzxN (source : Word64N) (sourceWidth : Nat) : Word64N :=
  word64n (source.val % 2 ^ sourceWidth)

/-- ``movsx`` sign-extension from ``sourceWidth`` bits into the 64-bit word. -/
def movsxN (source : Word64N) (sourceWidth : Nat) : Word64N :=
  let value := source.val % 2 ^ sourceWidth
  word64n (if value < 2 ^ (sourceWidth - 1) then value
           else value + MOD64N - 2 ^ sourceWidth)

theorem movLowN_val (old : Word64N) (shift value : Nat)
    (hdvd : 2 ^ shift ∣ MOD64N) :
    (word64n ((old.val / 2 ^ shift) * 2 ^ shift + value % 2 ^ shift)).val
      = (old.val / 2 ^ shift) * 2 ^ shift + value % 2 ^ shift := by
  simp only [word64n]
  apply Nat.mod_eq_of_lt
  have hq : (old.val / 2 ^ shift) * 2 ^ shift ≤ old.val := Nat.div_mul_le_self _ _
  have hquotient : (old.val / 2 ^ shift) * 2 ^ shift < MOD64N :=
    Nat.lt_of_le_of_lt hq old.isLt
  obtain ⟨multiple, hmultiple⟩ := hdvd
  have hrewritten : (old.val / 2 ^ shift) * 2 ^ shift < multiple * 2 ^ shift := by
    rw [Nat.mul_comm multiple (2 ^ shift), ← hmultiple]
    exact hquotient
  have hquotient_lt : old.val / 2 ^ shift < multiple :=
    Nat.lt_of_mul_lt_mul_right hrewritten
  have hstep : (old.val / 2 ^ shift) + 1 ≤ multiple :=
    Nat.succ_le_of_lt hquotient_lt
  have hbound : (old.val / 2 ^ shift) * 2 ^ shift + 2 ^ shift ≤ MOD64N := by
    have hmul : ((old.val / 2 ^ shift) + 1) * 2 ^ shift ≤ multiple * 2 ^ shift :=
      Nat.mul_le_mul_right _ hstep
    have hsplit : ((old.val / 2 ^ shift) + 1) * 2 ^ shift
        = (old.val / 2 ^ shift) * 2 ^ shift + 2 ^ shift := by
      rw [Nat.add_mul, Nat.one_mul]
    rw [hsplit, Nat.mul_comm multiple (2 ^ shift), ← hmultiple] at hmul
    exact hmul
  have hmod : value % 2 ^ shift < 2 ^ shift := Nat.mod_lt value (Nat.two_pow_pos shift)
  exact Nat.lt_of_lt_of_le (Nat.add_lt_add_left hmod _) hbound

theorem mov8N_val (old : Word64N) (value : Nat) :
    (mov8N old value).val = (old.val / 256) * 256 + value % 256 :=
  movLowN_val old 8 value (Nat.pow_dvd_pow 2 (by omega))

theorem mov16N_val (old : Word64N) (value : Nat) :
    (mov16N old value).val = (old.val / 65536) * 65536 + value % 65536 :=
  movLowN_val old 16 value (Nat.pow_dvd_pow 2 (by omega))

theorem mov8N_preserves_upper_bits (old : Word64N) (value : Nat) :
    (mov8N old value).val / 256 = old.val / 256 ∧
    (mov8N old value).val % 256 = value % 256 := by
  rw [mov8N_val]
  exact ⟨by omega, by omega⟩

theorem mov16N_preserves_upper_bits (old : Word64N) (value : Nat) :
    (mov16N old value).val / 65536 = old.val / 65536 ∧
    (mov16N old value).val % 65536 = value % 65536 := by
  rw [mov16N_val]
  exact ⟨by omega, by omega⟩

theorem movzx32N_zero_extension (source : Word64N) :
    (movzxN source 32).val = source.val % 4294967296 ∧
    (movzxN source 32).val < 4294967296 := by
  have hdvd : 2 ^ 32 ∣ MOD64N := Nat.pow_dvd_pow 2 (by omega)
  have hle : 2 ^ 32 ≤ MOD64N := Nat.le_of_dvd mod64n_pos hdvd
  have hbound : source.val % 2 ^ 32 < 2 ^ 32 := Nat.mod_lt _ (by omega)
  constructor
  · show (source.val % 2 ^ 32) % MOD64N = source.val % 2 ^ 32
    exact Nat.mod_eq_of_lt (Nat.lt_of_lt_of_le hbound hle)
  · show (source.val % 2 ^ 32) % MOD64N < 4294967296
    rw [Nat.mod_eq_of_lt (Nat.lt_of_lt_of_le hbound hle)]
    exact hbound

theorem movsx32N_sign_extension (source : Word64N) :
    (2 ^ 63 ≤ (movsxN source 32).val) ↔ 2 ^ 31 ≤ source.val % 2 ^ 32 := by
  have hM : MOD64N = 18446744073709551616 := rfl
  have hle : 2 ^ 32 ≤ MOD64N :=
    Nat.le_of_dvd mod64n_pos (Nat.pow_dvd_pow 2 (by omega))
  have hbound : source.val % 2 ^ 32 < 2 ^ 32 := Nat.mod_lt _ (by omega)
  by_cases hsign : source.val % 2 ^ 32 < 2 ^ 31
  · have hval : (movsxN source 32).val = source.val % 2 ^ 32 := by
      show (if source.val % 2 ^ 32 < 2 ^ 31
            then source.val % 2 ^ 32
            else source.val % 2 ^ 32 + MOD64N - 2 ^ 32) % MOD64N
        = source.val % 2 ^ 32
      rw [ite_eq_left hsign]
      exact Nat.mod_eq_of_lt (Nat.lt_of_lt_of_le hbound hle)
    constructor
    · intro h
      rw [hval] at h
      have h31 : 2 ^ 31 = 2147483648 := rfl
      have h63 : 2 ^ 63 = 9223372036854775808 := rfl
      omega
    · intro h
      have h31 : 2 ^ 31 = 2147483648 := rfl
      omega
  · have hval : (movsxN source 32).val
      = source.val % 2 ^ 32 + MOD64N - 2 ^ 32 := by
      show (if source.val % 2 ^ 32 < 2 ^ 31
            then source.val % 2 ^ 32
            else source.val % 2 ^ 32 + MOD64N - 2 ^ 32) % MOD64N
        = source.val % 2 ^ 32 + MOD64N - 2 ^ 32
      rw [ite_eq_right hsign]
      refine Nat.mod_eq_of_lt ?_
      have h32 : 2 ^ 32 = 4294967296 := rfl
      rw [hM, h32] at *
      omega
    constructor
    · intro h
      rw [hval] at h
      have h31 : 2 ^ 31 = 2147483648 := rfl
      have h63 : 2 ^ 63 = 9223372036854775808 := rfl
      have h32 : 2 ^ 32 = 4294967296 := rfl
      omega
    · intro h
      rw [hval]
      have h31 : 2 ^ 31 = 2147483648 := rfl
      have h63 : 2 ^ 63 = 9223372036854775808 := rfl
      have h32 : 2 ^ 32 = 4294967296 := rfl
      omega

/-! ## State transformers with exact flag and memory semantics

Every transformer either completes and returns a ``running`` state with
``stepIndex`` advanced, or fails closed into an explicit non-running status
(``boundExceeded`` for stack-window violations, ``exceptional`` for a
conditional branch that would read an undefined flag).  This mirrors the
Python reference, where the same conditions raise ``ValueError``. -/

def withResultN (state : MachineStateN) (index : Fin 16) (result : Word64N)
    (flags : FlagsN) : MachineStateN :=
  { state with
    gprs := fun i => if i = index then result else state.gprs i
    flags := flags
    stepIndex := state.stepIndex + 1 }

/-- ``add`` with writeback and exact ``_set_arithmetic_flags`` semantics. -/
def add64N (state : MachineStateN) (index : Fin 16) (value : Word64N) : MachineStateN :=
  withResultN state index (word64n ((state.gprs index).val + value.val))
    (addFlagsN (state.gprs index) value)

/-- ``sub`` with writeback and exact ``_set_arithmetic_flags`` semantics. -/
def sub64N (state : MachineStateN) (index : Fin 16) (value : Word64N) : MachineStateN :=
  withResultN state index
    (word64n ((state.gprs index).val + MOD64N - value.val))
    (subFlagsN (state.gprs index) value)

/-- ``and`` with writeback and exact ``_set_logic_flags`` semantics. -/
def and64N (state : MachineStateN) (index : Fin 16) (value : Word64N) : MachineStateN :=
  withResultN state index (word64n (Nat.land (state.gprs index).val value.val))
    (logicFlagsN (word64n (Nat.land (state.gprs index).val value.val)))

/-- ``or`` with writeback and exact ``_set_logic_flags`` semantics. -/
def or64N (state : MachineStateN) (index : Fin 16) (value : Word64N) : MachineStateN :=
  withResultN state index (word64n (Nat.lor (state.gprs index).val value.val))
    (logicFlagsN (word64n (Nat.lor (state.gprs index).val value.val)))

/-- ``xor`` with writeback and exact ``_set_logic_flags`` semantics. -/
def xor64N (state : MachineStateN) (index : Fin 16) (value : Word64N) : MachineStateN :=
  withResultN state index (word64n (Nat.xor (state.gprs index).val value.val))
    (logicFlagsN (word64n (Nat.xor (state.gprs index).val value.val)))

/-- ``cmp``: exactly the ``sub`` flag outcome with no register writeback. -/
def cmp64N (state : MachineStateN) (index : Fin 16) (value : Word64N) : MachineStateN :=
  { state with flags := subFlagsN (state.gprs index) value,
               stepIndex := state.stepIndex + 1 }

theorem add64N_flags (state : MachineStateN) (index : Fin 16) (value : Word64N) :
    (add64N state index value).flags = addFlagsN (state.gprs index) value := by
  simp [add64N, withResultN]

theorem sub64N_flags (state : MachineStateN) (index : Fin 16) (value : Word64N) :
    (sub64N state index value).flags = subFlagsN (state.gprs index) value := by
  simp [sub64N, withResultN]

theorem cmp64N_preserves_registers_and_memory (state : MachineStateN)
    (index : Fin 16) (value : Word64N) :
    (cmp64N state index value).gprs = state.gprs ∧
    (cmp64N state index value).memory = state.memory ∧
    (cmp64N state index value).rip = state.rip ∧
    (cmp64N state index value).flags = subFlagsN (state.gprs index) value := by
  simp only [cmp64N]
  exact ⟨trivial, trivial, trivial, trivial⟩

theorem sub64N_result_val (state : MachineStateN) (index : Fin 16) (value : Word64N) :
    ((sub64N state index value).gprs index).val
      = ((state.gprs index).val + MOD64N - value.val) % MOD64N := by
  simp [sub64N, withResultN, word64n]

/-- Adding a value and then subtracting the same value restores the 64-bit
register exactly: the two modular operations are inverse. -/
theorem word64n_add_sub_inverse_val (left value : Word64N) :
    (word64n ((word64n (left.val + value.val)).val + MOD64N - value.val)).val
      = left.val := by
  have hv := value.isLt
  have hl := left.isLt
  have hvle : value.val ≤ MOD64N := by omega
  show ((left.val + value.val) % MOD64N + MOD64N - value.val) % MOD64N = left.val
  rw [Nat.add_sub_assoc hvle]
  rw [Nat.mod_add_mod]
  have hsum : left.val + value.val + (MOD64N - value.val) = left.val + MOD64N := by
    omega
  rw [hsum, Nat.add_comm, Nat.add_mod_left]
  exact Nat.mod_eq_of_lt hl

theorem word64n_add_sub_inverse (left value : Word64N) :
    word64n ((word64n (left.val + value.val)).val + MOD64N - value.val) = left :=
  Fin.ext (word64n_add_sub_inverse_val left value)

theorem add_sub_inverse_gprs (state : MachineStateN) (index : Fin 16)
    (value : Word64N) :
    (sub64N (add64N state index value) index value).gprs = state.gprs := by
  funext i
  by_cases h : i = index
  · subst h
    have hinner : (add64N state i value).gprs i
        = word64n ((state.gprs i).val + value.val) := by
      simp [add64N, withResultN]
    show (sub64N (add64N state i value) i value).gprs i
      = state.gprs i
    simp only [sub64N, withResultN, hinner]
    exact word64n_add_sub_inverse _ _
  · show (sub64N (add64N state index value) index value).gprs i = state.gprs i
    simp [sub64N, add64N, withResultN, h]

/-! ## Bounded little-endian memory: byte order

``writeLittleEndianN``/``readLittleEndianN`` mirror ``_write_memory`` and
``_read_memory``: little-endian byte order over a bounded width. -/

def writeLittleEndianN : Nat → Nat → Nat → (Nat → UInt8) → (Nat → UInt8)
  | 0, _, _, memory => memory
  | width + 1, address, value, memory =>
      let lower := writeLittleEndianN width (address + 1) (value / 256) memory
      fun a => if a = address then UInt8.ofNat (value % 256) else lower a

def readLittleEndianN : Nat → Nat → (Nat → UInt8) → Nat
  | 0, _, _ => 0
  | width + 1, address, memory =>
      readLittleEndianN width (address + 1) memory * 256
        + UInt8.toNat (memory address)

theorem readLittleEndianN_congr (width address : Nat) (f g : Nat → UInt8)
    (hagreement : ∀ j, j < width → f (address + j) = g (address + j)) :
    readLittleEndianN width address f = readLittleEndianN width address g := by
  induction width generalizing address with
  | zero => rfl
  | succ width ih =>
      show readLittleEndianN width (address + 1) f * 256
        + UInt8.toNat (f address)
        = readLittleEndianN width (address + 1) g * 256
          + UInt8.toNat (g address)
      have h0 : f address = g address := by
        have hzero := hagreement 0 (by omega)
        rwa [Nat.add_zero] at hzero
      rw [ih (address + 1) (fun j hj => by
        have hshift : address + 1 + j = address + (j + 1) := by omega
        rw [hshift]
        exact hagreement (j + 1) (by omega)), h0]

/-- Modular fact used by the little-endian roundtrip: reading back a
little-endian write reassembles the written value. -/
theorem mulAddMod256_pow2 (q W r : Nat) (hW : 0 < W) (hr : r < 256) :
    (q % W) * 256 + r = (256 * q + r) % (256 * W) := by
  have hkey : (q % W) * 256 + r < 256 * W := by
    have hlt : q % W < W := Nat.mod_lt q hW
    omega
  have hkey2 : 256 * (q % W) + r < 256 * W := by
    have hlt : q % W < W := Nat.mod_lt q hW
    omega
  have hbase : W * (q / W) + q % W = q := Nat.div_add_mod q W
  have hsplit : 256 * W * (q / W) + 256 * (q % W) = 256 * q := by
    rw [Nat.mul_assoc, ← Nat.mul_add, hbase]
  have hcongr : (256 * q + r) % (256 * W) = (256 * (q % W) + r) % (256 * W) := by
    calc (256 * q + r) % (256 * W)
        = (256 * W * (q / W) + (256 * (q % W) + r)) % (256 * W) := by
          rw [← hsplit, Nat.add_assoc]
      _ = (256 * (q % W) + r) % (256 * W) := by
          rw [Nat.add_comm (256 * W * (q / W)) (256 * (q % W) + r)]
          exact Nat.add_mul_mod_self_left _ _ _
  rw [hcongr, Nat.mod_eq_of_lt hkey2, Nat.mul_comm (q % W) 256]

theorem read_write_little_endian_roundtrip (width address value : Nat)
    (memory : Nat → UInt8) :
    readLittleEndianN width address (writeLittleEndianN width address value memory)
      = value % 2 ^ (8 * width) := by
  induction width generalizing address value memory with
  | zero =>
      show 0 = value % 1
      exact (Nat.mod_one value).symm
  | succ width ih =>
      show readLittleEndianN width (address + 1)
            (fun a => if a = address then UInt8.ofNat (value % 256)
              else writeLittleEndianN width (address + 1) (value / 256) memory a)
            * 256
          + UInt8.toNat (if address = address then UInt8.ofNat (value % 256)
              else writeLittleEndianN width (address + 1) (value / 256) memory address)
          = value % 2 ^ (8 * (width + 1))
      have hbyte : UInt8.toNat (if address = address then UInt8.ofNat (value % 256)
          else writeLittleEndianN width (address + 1) (value / 256) memory address)
          = value % 256 := by
        rw [ite_eq_left rfl]
        simp
      have hagree : ∀ j, j < width →
          (fun a => if a = address then UInt8.ofNat (value % 256)
            else writeLittleEndianN width (address + 1) (value / 256) memory a)
            (address + 1 + j)
          = writeLittleEndianN width (address + 1) (value / 256) memory
            (address + 1 + j) := by
        intro j hj
        show (if address + 1 + j = address then UInt8.ofNat (value % 256)
          else writeLittleEndianN width (address + 1) (value / 256) memory
            (address + 1 + j))
          = writeLittleEndianN width (address + 1) (value / 256) memory
            (address + 1 + j)
        rw [ite_eq_right (by omega : ¬ (address + 1 + j = address))]
      rw [readLittleEndianN_congr width (address + 1) _ _ hagree, hbyte, ih]
      have hpow : 2 ^ (8 * (width + 1)) = 256 * 2 ^ (8 * width) := by
        have hexp : 8 * (width + 1) = 8 * width + 8 := by omega
        rw [hexp, Nat.pow_add]
        exact Nat.mul_comm _ _
      rw [hpow]
      have hstep : value % (256 * 2 ^ (8 * width))
          = (256 * (value / 256) + value % 256) % (256 * 2 ^ (8 * width)) :=
        congrArg (fun x => x % (256 * 2 ^ (8 * width)))
          (Nat.div_add_mod value 256).symm
      rw [hstep]
      exact mulAddMod256_pow2 (value / 256) (2 ^ (8 * width)) (value % 256)
        (Nat.two_pow_pos _) (Nat.mod_lt _ (by omega))

/-! ### Bounded stack window

``push64N``/``pop64N`` mirror the Python push/pop steps including the
``_check_stack_range`` fail-closed bound on the declared stack window. -/

def pushInBoundsN (state : MachineStateN) (index : Fin 16) (newRspValue : Nat) :
    MachineStateN :=
  { state with
    gprs := fun i => if i = rspIndexN then word64n newRspValue else state.gprs i
    memory := writeLittleEndianN 8 newRspValue (state.gprs index).val state.memory
    stepIndex := state.stepIndex + 1
    status := MachineStatusN.running }

def push64N (state : MachineStateN) (index : Fin 16) : MachineStateN :=
  if state.stackLower ≤ ((state.gprs rspIndexN).val + MOD64N - 8) % MOD64N ∧
      ((state.gprs rspIndexN).val + MOD64N - 8) % MOD64N + 8 ≤ state.stackUpper then
    pushInBoundsN state index (((state.gprs rspIndexN).val + MOD64N - 8) % MOD64N)
  else
    { state with status := MachineStatusN.boundExceeded }

def popInBoundsN (state : MachineStateN) (index : Fin 16) (value : Nat) :
    MachineStateN :=
  { state with
    gprs := fun i =>
      if i = rspIndexN then word64n ((state.gprs rspIndexN).val + 8)
      else if i = index then word64n value else state.gprs i
    stepIndex := state.stepIndex + 1
    status := MachineStatusN.running }

def pop64N (state : MachineStateN) (index : Fin 16) : MachineStateN :=
  if state.stackLower ≤ (state.gprs rspIndexN).val ∧
      (state.gprs rspIndexN).val + 8 ≤ state.stackUpper then
    popInBoundsN state index
      (readLittleEndianN 8 (state.gprs rspIndexN).val state.memory)
  else
    { state with status := MachineStatusN.boundExceeded }

theorem push64N_out_of_bounds_is_explicit (state : MachineStateN) (index : Fin 16)
    (hout : ¬ (state.stackLower ≤ ((state.gprs rspIndexN).val + MOD64N - 8) % MOD64N ∧
      ((state.gprs rspIndexN).val + MOD64N - 8) % MOD64N + 8 ≤ state.stackUpper)) :
    (push64N state index).status = MachineStatusN.boundExceeded ∧
    (push64N state index).gprs = state.gprs ∧
    (push64N state index).memory = state.memory := by
  simp only [push64N, ite_eq_right hout]
  exact ⟨trivial, trivial, trivial⟩

theorem pop64N_out_of_bounds_is_explicit (state : MachineStateN) (index : Fin 16)
    (hout : ¬ (state.stackLower ≤ (state.gprs rspIndexN).val ∧
      (state.gprs rspIndexN).val + 8 ≤ state.stackUpper)) :
    (pop64N state index).status = MachineStatusN.boundExceeded ∧
    (pop64N state index).gprs = state.gprs ∧
    (pop64N state index).memory = state.memory := by
  simp only [pop64N, ite_eq_right hout]
  exact ⟨trivial, trivial, trivial⟩

/-- Push into the bounded window then pop back restores the stack pointer and
the pushed register, and the pop observes the exact pushed value. -/
theorem push64N_pop64N_roundtrip (state : MachineStateN) (index : Fin 16)
    (hsp : 8 ≤ (state.gprs rspIndexN).val)
    (hstack : state.stackLower + 8 ≤ (state.gprs rspIndexN).val ∧
      (state.gprs rspIndexN).val + 8 ≤ state.stackUpper)
    (hne : index ≠ rspIndexN) :
    (pop64N (push64N state index) index).gprs index = state.gprs index ∧
    (pop64N (push64N state index) index).gprs rspIndexN = state.gprs rspIndexN ∧
    (pop64N (push64N state index) index).status = MachineStatusN.running := by
  have hM16 := (state.gprs rspIndexN).isLt
  have hMindex := (state.gprs index).isLt
  have hM : MOD64N = 18446744073709551616 := rfl
  have hnewLt : (state.gprs rspIndexN).val - 8 < MOD64N := by omega
  have hmod : ((state.gprs rspIndexN).val + MOD64N - 8) % MOD64N
      = (state.gprs rspIndexN).val - 8 := by
    have hsplit : (state.gprs rspIndexN).val + MOD64N - 8
        = (state.gprs rspIndexN).val - 8 + MOD64N := by omega
    rw [hsplit, Nat.add_comm, Nat.add_mod_left]
    exact Nat.mod_eq_of_lt hnewLt
  have hpushCond : state.stackLower ≤ ((state.gprs rspIndexN).val + MOD64N - 8) % MOD64N ∧
      ((state.gprs rspIndexN).val + MOD64N - 8) % MOD64N + 8 ≤ state.stackUpper := by
    rw [hmod]
    exact ⟨by omega, by omega⟩
  have hpopCond : state.stackLower ≤ (state.gprs rspIndexN).val - 8 ∧
      (state.gprs rspIndexN).val - 8 + 8 ≤ state.stackUpper := by omega
  have hpush : push64N state index
      = pushInBoundsN state index (((state.gprs rspIndexN).val + MOD64N - 8) % MOD64N) := by
    simp only [push64N]
    exact ite_eq_left hpushCond
  have hpushRsp :
      ((pushInBoundsN state index
        (((state.gprs rspIndexN).val + MOD64N - 8) % MOD64N)).gprs rspIndexN).val
      = (state.gprs rspIndexN).val - 8 := by
    show ((if rspIndexN = rspIndexN
      then word64n (((state.gprs rspIndexN).val + MOD64N - 8) % MOD64N)
      else state.gprs rspIndexN)).val = (state.gprs rspIndexN).val - 8
    rw [ite_eq_left rfl]
    show ((state.gprs rspIndexN).val + MOD64N - 8) % MOD64N % MOD64N
      = (state.gprs rspIndexN).val - 8
    rw [Nat.mod_mod]
    exact hmod
  have hread : readLittleEndianN 8 ((state.gprs rspIndexN).val - 8)
      (pushInBoundsN state index
        (((state.gprs rspIndexN).val + MOD64N - 8) % MOD64N)).memory
      = (state.gprs index).val := by
    show readLittleEndianN 8 ((state.gprs rspIndexN).val - 8)
      (writeLittleEndianN 8 (((state.gprs rspIndexN).val + MOD64N - 8) % MOD64N)
        (state.gprs index).val state.memory) = (state.gprs index).val
    rw [hmod, read_write_little_endian_roundtrip]
    show (state.gprs index).val % 2 ^ 64 = (state.gprs index).val
    rw [show (2 ^ 64 : Nat) = MOD64N from rfl]
    exact Nat.mod_eq_of_lt hMindex
  have hpop : pop64N (pushInBoundsN state index
        (((state.gprs rspIndexN).val + MOD64N - 8) % MOD64N)) index
      = popInBoundsN (pushInBoundsN state index
        (((state.gprs rspIndexN).val + MOD64N - 8) % MOD64N)) index
        (state.gprs index).val := by
    simp only [pop64N]
    rw [hpushRsp, hread]
    exact ite_eq_left hpopCond
  rw [hpush, hpop]
  refine ⟨?_, ?_, rfl⟩
  · show ((if index = rspIndexN then
        word64n (((pushInBoundsN state index
          (((state.gprs rspIndexN).val + MOD64N - 8) % MOD64N)).gprs rspIndexN).val + 8)
      else if index = index then word64n ((state.gprs index).val)
      else (pushInBoundsN state index
        (((state.gprs rspIndexN).val + MOD64N - 8) % MOD64N)).gprs index)) = state.gprs index
    rw [ite_eq_right hne, ite_eq_left rfl]
    apply Fin.ext
    show ((state.gprs index).val) % MOD64N = (state.gprs index).val
    exact Nat.mod_eq_of_lt hMindex
  · show ((if rspIndexN = rspIndexN
        then word64n (((pushInBoundsN state index
          (((state.gprs rspIndexN).val + MOD64N - 8) % MOD64N)).gprs rspIndexN).val + 8)
        else state.gprs rspIndexN)) = state.gprs rspIndexN
    rw [ite_eq_left rfl, hpushRsp]
    have hadd : (state.gprs rspIndexN).val - 8 + 8 = (state.gprs rspIndexN).val := by
      omega
    rw [hadd]
    apply Fin.ext
    show ((state.gprs rspIndexN).val) % MOD64N = (state.gprs rspIndexN).val
    exact Nat.mod_eq_of_lt hM16

/-! ## Conditional branches over the definedness map

``jccHoldsN`` mirrors ``_condition_holds``: every flag read goes through the
definedness map, and an undefined flag read produces ``none`` which
``jccStateN`` turns into the explicit ``exceptional`` status (the Python
reference raises ``ValueError`` for the same condition). -/

inductive JccConditionN where
  | e | ne | s | ns | l | ge | le | g | c | nc | o | no | p | np
deriving Repr, DecidableEq

def flagValueAtBitN (flags : FlagsN) (bit : Nat) : Bool :=
  if bit = cfBitN then flags.cf
  else if bit = pfBitN then flags.pf
  else if bit = afBitN then flags.af
  else if bit = zfBitN then flags.zf
  else if bit = sfBitN then flags.sf
  else flags.of

def readFlagN (state : MachineStateN) (bit : Nat) : Option Bool :=
  if state.flags.defined bit then some (flagValueAtBitN state.flags bit) else none

def jccHoldsN (state : MachineStateN) (condition : JccConditionN) : Option Bool :=
  match condition with
  | .e => readFlagN state zfBitN
  | .ne => (readFlagN state zfBitN).map not
  | .s => readFlagN state sfBitN
  | .ns => (readFlagN state sfBitN).map not
  | .c => readFlagN state cfBitN
  | .nc => (readFlagN state cfBitN).map not
  | .o => readFlagN state ofBitN
  | .no => (readFlagN state ofBitN).map not
  | .p => readFlagN state pfBitN
  | .np => (readFlagN state pfBitN).map not
  | .l =>
      match readFlagN state sfBitN, readFlagN state ofBitN with
      | some sf, some ofValue => some (sf != ofValue)
      | _, _ => none
  | .ge =>
      match readFlagN state sfBitN, readFlagN state ofBitN with
      | some sf, some ofValue => some (sf == ofValue)
      | _, _ => none
  | .le =>
      match readFlagN state zfBitN, readFlagN state sfBitN,
          readFlagN state ofBitN with
      | some zf, some sf, some ofValue => some (zf || sf != ofValue)
      | _, _, _ => none
  | .g =>
      match readFlagN state zfBitN, readFlagN state sfBitN,
          readFlagN state ofBitN with
      | some zf, some sf, some ofValue => some (!zf && sf == ofValue)
      | _, _, _ => none

def jccStateN (state : MachineStateN) (condition : JccConditionN)
    (taken fallthrough : Nat) : MachineStateN :=
  match jccHoldsN state condition with
  | some decision =>
      if decision then
        { state with rip := word64n taken, stepIndex := state.stepIndex + 1 }
      else
        { state with rip := word64n fallthrough, stepIndex := state.stepIndex + 1 }
  | none => { state with status := MachineStatusN.exceptional }

/-- A machine state identical to ``state`` except for a fully-defined zero
flag, used to witness taken vs not-taken branch distinctness. -/
def stateWithZfN (state : MachineStateN) (zf : Bool) : MachineStateN :=
  { state with flags := { state.flags with zf := zf, defined := fun _ => true } }

/-- A fully-defined zero flag decides the branch: taken sets the branch target,
not-taken sets the fallthrough, and the two outcomes are distinct states. -/
theorem jcc_zero_flag_branch_distinct (state : MachineStateN) (taken fallthrough : Nat)
    (hne : word64n taken ≠ word64n fallthrough) :
    jccStateN (stateWithZfN state true) JccConditionN.e taken fallthrough ≠
      jccStateN (stateWithZfN state false) JccConditionN.e taken fallthrough := by
  intro hEq
  have htrue : jccHoldsN (stateWithZfN state true) JccConditionN.e = some true := by
    simp [jccHoldsN, readFlagN, flagValueAtBitN, stateWithZfN, zfBitN, cfBitN,
      pfBitN, afBitN]
  have hfalse : jccHoldsN (stateWithZfN state false) JccConditionN.e = some false := by
    simp [jccHoldsN, readFlagN, flagValueAtBitN, stateWithZfN, zfBitN, cfBitN,
      pfBitN, afBitN]
  simp only [jccStateN, htrue, hfalse, ite_true] at hEq
  exact hne (congrArg MachineStateN.rip hEq)

/-- Reading a conditional branch over an undefined zero flag fails closed into
the explicit ``exceptional`` status, mirroring the Python ``ValueError``. -/
theorem jcc_undefined_flag_is_exceptional (state : MachineStateN)
    (taken fallthrough : Nat) (hundef : state.flags.defined zfBitN = false) :
    (jccStateN state JccConditionN.e taken fallthrough).status
      = MachineStatusN.exceptional := by
  have hholds : jccHoldsN state JccConditionN.e = none := by
    show (if state.flags.defined zfBitN = true then
        some (flagValueAtBitN state.flags zfBitN) else none) = none
    rw [hundef]
    simp
  simp only [jccStateN, hholds]

/-- When the zero flag is defined and set, the taken branch target is installed. -/
theorem jcc_e_taken_sets_rip (state : MachineStateN) (taken fallthrough : Nat)
    (hdefined : state.flags.defined zfBitN = true) (hzf : state.flags.zf = true) :
    (jccStateN state JccConditionN.e taken fallthrough).rip = word64n taken := by
  have hholds : jccHoldsN state JccConditionN.e = some true := by
    show (if state.flags.defined zfBitN = true then
        some (flagValueAtBitN state.flags zfBitN) else none) = some true
    rw [hdefined]
    show (some state.flags.zf) = some true
    rw [hzf]
  simp only [jccStateN, hholds]
  rfl

/-- When the zero flag is defined and clear, the fallthrough target is installed. -/
theorem jcc_e_not_taken_sets_rip (state : MachineStateN) (taken fallthrough : Nat)
    (hdefined : state.flags.defined zfBitN = true) (hzf : state.flags.zf = false) :
    (jccStateN state JccConditionN.e taken fallthrough).rip = word64n fallthrough := by
  have hholds : jccHoldsN state JccConditionN.e = some false := by
    show (if state.flags.defined zfBitN = true then
        some (flagValueAtBitN state.flags zfBitN) else none) = some false
    rw [hdefined]
    show (some state.flags.zf) = some false
    rw [hzf]
  simp only [jccStateN, hholds]
  rfl

end ActiveAssembly
