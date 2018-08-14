#include <ostream>
#include <iostream>

#ifndef GDB_DEBUG_H_
#define GDB_DEBUG_H_

extern int display_gdb_output;

namespace GdbColor {
    enum Code {
        FG_RED      = 31,
        FG_GREEN    = 32,
        FG_BLUE     = 34,
        FG_DEFAULT  = 39,
        BG_RED      = 41,
        BG_GREEN    = 42,
        BG_BLUE     = 44,
        BG_DEFAULT  = 49
    };
    std::ostream& operator<<(std::ostream& os, Code code);
}


//class ColorOut: public std::ostream
class ColorOut
{
private:
    std::ostream& out;
    GdbColor::Code color;
public:
    ColorOut(std::ostream& _out, GdbColor::Code _color):out(_out), color(_color){}

    ~ColorOut() {}

    template<typename T> 
    const ColorOut& operator<< (const T& val) const
    {
        out << "\x1b[" << std::to_string(color) << ";49m" << val << "\x1b[39;49m"; //GdbColor::FG_DEFAULT;
        return *this;
    }
    /* don't color stream manipulators */
    const ColorOut& operator<< (std::ostream& (*pf)(std::ostream&)) const 
    {
        out << pf;
        return *this;
    }
};

static ColorOut dout(std::cout, GdbColor::FG_RED);
static ColorOut sout(std::cout, GdbColor::FG_GREEN);
static ColorOut hout(std::cout, GdbColor::FG_BLUE);


#endif /*GDB_DEBUG_H_*/
