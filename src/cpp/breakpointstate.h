#ifndef BREAKPOINTSTATE_H
#define BREAKPOINTSTATE_H

enum class BreakpointState : int {
    UNDEFINED = -1,
    NONE = 0,
    UPPER,
    LOWER,
    TIME
};

#endif
