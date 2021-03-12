#ifndef YAK_MARCHING_CUBES_TABLES_H
#define YAK_MARCHING_CUBES_TABLES_H

namespace yak {
extern const int edgeFlags[256];

extern const int triangleTable[256][16];

extern const int edgeConnections[12][2];

extern const int numVertsTable[256];

} // namespace yak

#endif
