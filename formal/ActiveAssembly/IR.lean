/-!
# Normalized instruction intermediate representation

The IR is a closed-world vocabulary shared conceptually with the Python
receipt layer.  It names the instruction families that the research protocol
may contain and keeps unsupported effects explicit instead of silently mapping
them to a no-op.  This file is a schema and classification boundary; it is
not, by itself, a theorem about native x86 execution.
-/

namespace ActiveAssembly

inductive OperandWidthN where
  | w8
  | w16
  | w32
  | w64
deriving Repr, DecidableEq

inductive FlagN where
  | cf
  | pf
  | af
  | zf
  | sf
  | of
deriving Repr, DecidableEq

inductive OpcodeN where
  | mov
  | movabs
  | movzx
  | movsx
  | lea
  | add
  | adc
  | sub
  | sbb
  | neg
  | and
  | or
  | xor
  | not
  | cmp
  | test
  | inc
  | dec
  | shl
  | shr
  | sar
  | nop
  | jmp
  | conditionalBranch
  | push
  | pop
  | call
  | ret
  | unsupported
deriving Repr, DecidableEq

inductive ControlFlowN where
  | fallthrough
  | branch
  | call
  | return
  | unsupported
deriving Repr, DecidableEq

structure MemoryAddressN where
  base : Option (Fin 16)
  index : Option (Fin 16)
  scale : Nat
  displacement : Int
deriving Repr, DecidableEq

structure IRInstructionN where
  opcode : OpcodeN
  width : OperandWidthN
  immediate : Option Int
  memoryAddress : Option MemoryAddressN
  flagReads : List FlagN
  flagWrites : List FlagN
  controlFlow : ControlFlowN
deriving Repr, DecidableEq

def OpcodeN.isDeclaredSupported : OpcodeN → Bool
  | .unsupported => false
  | _ => true

theorem explicit_unsupported_is_not_supported :
    OpcodeN.isDeclaredSupported .unsupported = false := by
  rfl

def declared_opcode_decidable (opcode : OpcodeN) :
    Decidable (OpcodeN.isDeclaredSupported opcode = true) := by
  exact inferInstance

theorem declared_opcode_has_decidable_status (opcode : OpcodeN) :
    OpcodeN.isDeclaredSupported opcode = true ∨
      OpcodeN.isDeclaredSupported opcode = false := by
  cases opcode <;> simp [OpcodeN.isDeclaredSupported]

end ActiveAssembly
