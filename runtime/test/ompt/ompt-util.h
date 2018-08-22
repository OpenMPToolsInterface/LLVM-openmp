#ifndef _BSD_SOURCE
#define _BSD_SOURCE
#endif
#define _DEFAULT_SOURCE
#include <stdio.h>
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>
#include <omp.h>
#include <ompt.h>
#include "ompt-signal.h"

// Used to detect architecture
#include "../../src/kmp_platform.h"

static const char *ompt_thread_t_values[] = {
    NULL, "ompt_thread_initial", "ompt_thread_worker", "ompt_thread_other"};

static const char *ompt_task_status_t_values[] = {
    NULL, "ompt_task_complete", "ompt_task_yield", "ompt_task_cancel",
    "ompt_task_others"};
static const char *ompt_cancel_flag_t_values[] = {
    "ompt_cancel_parallel",      "ompt_cancel_sections",
    "ompt_cancel_loop",          "ompt_cancel_taskgroup",
    "ompt_cancel_activated",     "ompt_cancel_detected",
    "ompt_cancel_discarded_task"};

static void format_task_type(int type, char *buffer) {
  char *progress = buffer;
  if (type & ompt_task_initial)
    progress += sprintf(progress, "ompt_task_initial");
  if (type & ompt_task_implicit)
    progress += sprintf(progress, "ompt_task_implicit");
  if (type & ompt_task_explicit)
    progress += sprintf(progress, "ompt_task_explicit");
  if (type & ompt_task_target)
    progress += sprintf(progress, "ompt_task_target");
  if (type & ompt_task_undeferred)
    progress += sprintf(progress, "|ompt_task_undeferred");
  if (type & ompt_task_untied)
    progress += sprintf(progress, "|ompt_task_untied");
  if (type & ompt_task_final)
    progress += sprintf(progress, "|ompt_task_final");
  if (type & ompt_task_mergeable)
    progress += sprintf(progress, "|ompt_task_mergeable");
  if (type & ompt_task_merged)
    progress += sprintf(progress, "|ompt_task_merged");
}

static ompt_set_callback_t ompt_set_callback;
static ompt_get_callback_t ompt_get_callback;
static ompt_get_state_t ompt_get_state;
static ompt_get_task_info_t ompt_get_task_info;
static ompt_get_thread_data_t ompt_get_thread_data;
static ompt_get_parallel_info_t ompt_get_parallel_info;
static ompt_get_unique_id_t ompt_get_unique_id;
static ompt_get_num_procs_t ompt_get_num_procs;
static ompt_get_num_places_t ompt_get_num_places;
static ompt_get_place_proc_ids_t ompt_get_place_proc_ids;
static ompt_get_place_num_t ompt_get_place_num;
static ompt_get_partition_place_nums_t ompt_get_partition_place_nums;
static ompt_get_proc_id_t ompt_get_proc_id;
static ompt_enumerate_states_t ompt_enumerate_states;
static ompt_enumerate_mutex_impls_t ompt_enumerate_mutex_impls;

static void print_ids(int level) {
  int task_type, thread_num;
  omp_frame_t *frame;
  ompt_data_t *task_parallel_data;
  ompt_data_t *task_data;
  int exists_task = ompt_get_task_info(level, &task_type, &task_data, &frame,
                                       &task_parallel_data, &thread_num);
  char buffer[2048];
  format_task_type(task_type, buffer);
  if (frame)
    printf("%" PRIu64 ": task level %d: parallel_id=%" PRIu64
           ", task_id=%" PRIu64 ", exit_frame=%p, reenter_frame=%p, "
           "task_type=%s=%d, thread_num=%d\n",
           ompt_get_thread_data()->value, level,
           exists_task ? task_parallel_data->value : 0,
           exists_task ? task_data->value : 0, frame->exit_frame,
           frame->enter_frame, buffer, task_type, thread_num);
}

#define get_frame_address(level) __builtin_frame_address(level)

#define print_frame(level)                                                     \
  printf("%" PRIu64 ": __builtin_frame_address(%d)=%p\n",                      \
         ompt_get_thread_data()->value, level, get_frame_address(level))

// clang (version 5.0 and above) adds an intermediate function call with debug
// flag (-g)
#if defined(TEST_NEED_PRINT_FRAME_FROM_OUTLINED_FN)
#if defined(DEBUG) && defined(__clang__) && __clang_major__ >= 5
#define print_frame_from_outlined_fn(level) print_frame(level + 1)
#else
#define print_frame_from_outlined_fn(level) print_frame(level)
#endif

#if defined(__clang__) && __clang_major__ >= 5
#warning                                                                       \
    "Clang 5.0 and later add an additional wrapper for outlined functions when compiling with debug information."
#warning                                                                       \
    "Please define -DDEBUG iff you manually pass in -g to make the tests succeed!"
#endif
#endif

// This macro helps to define a label at the current position that can be used
// to get the current address in the code.
//
// For print_current_address():
//   To reliably determine the offset between the address of the label and the
//   actual return address, we insert a NOP instruction as a jump target as the
//   compiler would otherwise insert an instruction that we can't control. The
//   instruction length is target dependent and is explained below.
//
// (The empty block between "#pragma omp ..." and the __asm__ statement is a
// workaround for a bug in the Intel Compiler.)
#define define_ompt_label(id)                                                  \
  {}                                                                           \
  __asm__("nop");                                                              \
  ompt_label_##id:

// This macro helps to get the address of a label that is inserted by the above
// macro define_ompt_label(). The address is obtained with a GNU extension
// (&&label) that has been tested with gcc, clang and icc.
#define get_ompt_label_address(id) (&&ompt_label_##id)

// This macro prints the exact address that a previously called runtime function
// returns to.
#define print_current_address(id)                                              \
  define_ompt_label(id)                                                        \
      print_possible_return_addresses(get_ompt_label_address(id))

#if KMP_ARCH_X86 || KMP_ARCH_X86_64
// On X86 the NOP instruction is 1 byte long. In addition, the comiler inserts
// a MOV instruction for non-void runtime functions which is 3 bytes long.
#define print_possible_return_addresses(addr)                                  \
  printf("%" PRIu64 ": current_address=%p or %p for non-void functions\n",     \
         ompt_get_thread_data()->value, ((char *)addr) - 1,                    \
         ((char *)addr) - 4)
#elif KMP_ARCH_PPC64
// On Power the NOP instruction is 4 bytes long. In addition, the compiler
// inserts an LD instruction which accounts for another 4 bytes. In contrast to
// X86 this instruction is always there, even for void runtime functions.
#define print_possible_return_addresses(addr)                                  \
  printf("%" PRIu64 ": current_address=%p\n", ompt_get_thread_data()->value,   \
         ((char *)addr) - 8)
#elif KMP_ARCH_AARCH64
// On AArch64 the NOP instruction is 4 bytes long, can be followed by inserted
// store instruction (another 4 bytes long).
#define print_possible_return_addresses(addr)                                  \
  printf("%" PRIu64 ": current_address=%p or %p\n",                            \
         ompt_get_thread_data()->value, ((char *)addr) - 4,                    \
         ((char *)addr) - 8)
#else
#error Unsupported target architecture, cannot determine address offset!
#endif

// This macro performs a somewhat similar job to print_current_address(), except
// that it discards a certain number of nibbles from the address and only prints
// the most significant bits / nibbles. This can be used for cases where the
// return address can only be approximated.
//
// To account for overflows (ie the most significant bits / nibbles have just
// changed as we are a few bytes above the relevant power of two) the addresses
// of the "current" and of the "previous block" are printed.
#define print_fuzzy_address(id)                                                \
  define_ompt_label(id) print_fuzzy_address_blocks(get_ompt_label_address(id))

// If you change this define you need to adapt all capture patterns in the tests
// to include or discard the new number of nibbles!
#define FUZZY_ADDRESS_DISCARD_NIBBLES 2
#define FUZZY_ADDRESS_DISCARD_BYTES (1 << ((FUZZY_ADDRESS_DISCARD_NIBBLES)*4))
#define print_fuzzy_address_blocks(addr)                                       \
  printf("%" PRIu64 ": fuzzy_address=0x%" PRIx64 " or 0x%" PRIx64              \
         " or 0x%" PRIx64 " or 0x%" PRIx64 " (%p)\n",                          \
         ompt_get_thread_data()->value,                                        \
         ((uint64_t)addr) / FUZZY_ADDRESS_DISCARD_BYTES - 1,                   \
         ((uint64_t)addr) / FUZZY_ADDRESS_DISCARD_BYTES,                       \
         ((uint64_t)addr) / FUZZY_ADDRESS_DISCARD_BYTES + 1,                   \
         ((uint64_t)addr) / FUZZY_ADDRESS_DISCARD_BYTES + 2, addr)

#define register_callback_t(name, type)                                        \
  do {                                                                         \
    type f_##name = &on_##name;                                                \
    if (ompt_set_callback(name, (ompt_callback_t)f_##name) == ompt_set_never)  \
      printf("0: Could not register callback '" #name "'\n");                  \
  } while (0)

#define register_callback(name) register_callback_t(name, name##_t)
