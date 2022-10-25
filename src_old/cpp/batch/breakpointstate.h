#ifndef BREAKPOINTSTATE_H
#define BREAKPOINTSTATE_H

enum class BreakpointState : int {
    UNDEFINED = 0,
    NONE = 1,
    UPPER,
    LOWER,
    TIME
};

#endif
