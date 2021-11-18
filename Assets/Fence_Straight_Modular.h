// "Fence_Straight_Modular.h" generated by "Obj2Header.exe" [Version 1.9d] Author: L.Norri Fullsail University.
// Data is converted to left-handed coordinate system and UV data is V flipped for Direct3D/Vulkan.
// The companion file "Fence_Straight_Modular.h2b" is a binary dump of this format(-padding) for more flexibility.
// Loading *.h2b: read version, sizes, handle strings(max len 260) by reading until null-terminated.
/***********************/
/*  Generator version  */
/***********************/
#ifndef _Fence_Straight_Modular_version_
const char Fence_Straight_Modular_version[4] = { '0','1','9','d' };
#define _Fence_Straight_Modular_version_
#endif
/************************************************/
/*  This section contains the model's size data */
/************************************************/
#ifndef _Fence_Straight_Modular_vertexcount_
const unsigned Fence_Straight_Modular_vertexcount = 200;
#define _Fence_Straight_Modular_vertexcount_
#endif
#ifndef _Fence_Straight_Modular_indexcount_
const unsigned Fence_Straight_Modular_indexcount = 312;
#define _Fence_Straight_Modular_indexcount_
#endif
#ifndef _Fence_Straight_Modular_materialcount_
const unsigned Fence_Straight_Modular_materialcount = 2; // can be used for batched draws
#define _Fence_Straight_Modular_materialcount_
#endif
#ifndef _Fence_Straight_Modular_meshcount_
const unsigned Fence_Straight_Modular_meshcount = 2;
#define _Fence_Straight_Modular_meshcount_
#endif
/************************************************/
/*  This section contains the raw data to load  */
/************************************************/
#ifndef __OBJ_VEC3__
typedef struct _OBJ_VEC3_
{
	float x,y,z; // 3D Coordinate.
}OBJ_VEC3;
#define __OBJ_VEC3__
#endif
#ifndef __OBJ_VERT__
typedef struct _OBJ_VERT_
{
	OBJ_VEC3 pos; // Left-handed +Z forward coordinate w not provided, assumed to be 1.
	OBJ_VEC3 uvw; // D3D/Vulkan style top left 0,0 coordinate.
	OBJ_VEC3 nrm; // Provided direct from obj file, may or may not be normalized.
}OBJ_VERT;
#define __OBJ_VERT__
#endif
#ifndef _Fence_Straight_Modular_vertices_
// Raw Vertex Data follows: Positions, Texture Coordinates and Normals.
const OBJ_VERT Fence_Straight_Modular_vertices[200] =
{
	{	{ 0.273744f, -0.426092f, 0.273744f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ 0.273744f, -0.426092f, -0.273744f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ 0.206188f, -0.426092f, -0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ 0.206189f, -0.426092f, 0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ -0.273744f, -0.780422f, -0.273744f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -1.000000f, 0.000000f, -0.000000f }	},
	{	{ -0.273744f, -0.780422f, 0.273744f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -1.000000f, 0.000000f, -0.000000f }	},
	{	{ -0.273744f, -0.426092f, 0.273744f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -1.000000f, 0.000000f, -0.000000f }	},
	{	{ -0.273744f, -0.426092f, -0.273744f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -1.000000f, 0.000000f, -0.000000f }	},
	{	{ -0.273744f, -0.426092f, 0.273744f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ -0.206188f, -0.426092f, 0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ -0.273744f, -0.426092f, -0.273744f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ -0.206189f, -0.426092f, -0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ 0.273744f, -0.780422f, 0.273744f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 1.000000f, 0.000000f, -0.000000f }	},
	{	{ 0.273744f, -0.780422f, -0.273744f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 1.000000f, 0.000000f, -0.000000f }	},
	{	{ 0.273744f, -0.426092f, -0.273744f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 1.000000f, 0.000000f, -0.000000f }	},
	{	{ 0.273744f, -0.426092f, 0.273744f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 1.000000f, 0.000000f, -0.000000f }	},
	{	{ 0.273744f, -0.780422f, -0.273744f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, -1.000000f }	},
	{	{ -0.273744f, -0.780422f, -0.273744f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, -1.000000f }	},
	{	{ -0.273744f, -0.426092f, -0.273744f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, -1.000000f }	},
	{	{ 0.273744f, -0.426092f, -0.273744f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, -1.000000f }	},
	{	{ -0.273744f, -0.780422f, 0.273744f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, 1.000000f }	},
	{	{ 0.273744f, -0.780422f, 0.273744f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, 1.000000f }	},
	{	{ 0.273744f, -0.426092f, 0.273744f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, 1.000000f }	},
	{	{ -0.273744f, -0.426092f, 0.273744f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, 1.000000f }	},
	{	{ -0.390598f, -0.999089f, -0.390598f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -1.000000f, 0.000000f, -0.000000f }	},
	{	{ -0.390598f, -0.999089f, 0.390598f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -1.000000f, 0.000000f, -0.000000f }	},
	{	{ -0.390598f, -0.877727f, 0.390598f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -1.000000f, 0.000000f, -0.000000f }	},
	{	{ -0.390598f, -0.877727f, -0.390598f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -1.000000f, 0.000000f, -0.000000f }	},
	{	{ 0.390598f, -0.999089f, 0.390598f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 1.000000f, 0.000000f, -0.000000f }	},
	{	{ 0.390598f, -0.999089f, -0.390598f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 1.000000f, 0.000000f, -0.000000f }	},
	{	{ 0.390598f, -0.877727f, -0.390598f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 1.000000f, 0.000000f, -0.000000f }	},
	{	{ 0.390598f, -0.877727f, 0.390598f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 1.000000f, 0.000000f, -0.000000f }	},
	{	{ 0.390598f, -0.999089f, -0.390598f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, -1.000000f }	},
	{	{ -0.390598f, -0.999089f, -0.390598f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, -1.000000f }	},
	{	{ -0.390598f, -0.877727f, -0.390598f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, -1.000000f }	},
	{	{ 0.390598f, -0.877727f, -0.390598f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, -1.000000f }	},
	{	{ -0.390598f, -0.999089f, 0.390598f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, 1.000000f }	},
	{	{ 0.390598f, -0.999089f, 0.390598f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, 1.000000f }	},
	{	{ 0.390598f, -0.877727f, 0.390598f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, 1.000000f }	},
	{	{ -0.390598f, -0.877727f, 0.390598f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, 1.000000f }	},
	{	{ -0.390598f, -0.999089f, 0.390598f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -1.000000f, 0.000000f }	},
	{	{ -0.390598f, -0.999089f, -0.390598f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -1.000000f, 0.000000f }	},
	{	{ 0.390598f, -0.999089f, -0.390598f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -1.000000f, 0.000000f }	},
	{	{ 0.390598f, -0.999089f, 0.390598f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -1.000000f, 0.000000f }	},
	{	{ -0.369984f, -0.851557f, -0.369984f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.594400f, 0.804200f, -0.000000f }	},
	{	{ -0.369984f, -0.851557f, 0.369984f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.594400f, 0.804200f, -0.000000f }	},
	{	{ -0.273744f, -0.780422f, 0.273744f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.594400f, 0.804200f, -0.000000f }	},
	{	{ -0.273744f, -0.780422f, -0.273744f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.594400f, 0.804200f, -0.000000f }	},
	{	{ 0.369984f, -0.851557f, 0.369984f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.594400f, 0.804200f, 0.000000f }	},
	{	{ 0.369984f, -0.851557f, -0.369984f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.594400f, 0.804200f, 0.000000f }	},
	{	{ 0.273744f, -0.780422f, -0.273744f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.594400f, 0.804200f, 0.000000f }	},
	{	{ 0.273744f, -0.780422f, 0.273744f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.594400f, 0.804200f, 0.000000f }	},
	{	{ 0.369984f, -0.851557f, -0.369984f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.000000f, 0.804200f, -0.594400f }	},
	{	{ -0.369984f, -0.851557f, -0.369984f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.000000f, 0.804200f, -0.594400f }	},
	{	{ -0.273744f, -0.780422f, -0.273744f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.000000f, 0.804200f, -0.594400f }	},
	{	{ 0.273744f, -0.780422f, -0.273744f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.000000f, 0.804200f, -0.594400f }	},
	{	{ 0.369984f, -0.851557f, 0.369984f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.618800f, 0.785600f }	},
	{	{ -0.369984f, -0.851557f, 0.369984f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.618800f, 0.785600f }	},
	{	{ -0.390598f, -0.877727f, 0.390598f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.618800f, 0.785600f }	},
	{	{ 0.390598f, -0.877727f, 0.390598f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.618800f, 0.785600f }	},
	{	{ 0.369984f, -0.851557f, -0.369984f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.785600f, 0.618800f, -0.000000f }	},
	{	{ 0.369984f, -0.851557f, 0.369984f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.785600f, 0.618800f, -0.000000f }	},
	{	{ 0.390598f, -0.877727f, 0.390598f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.785600f, 0.618800f, -0.000000f }	},
	{	{ 0.390598f, -0.877727f, -0.390598f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.785600f, 0.618800f, -0.000000f }	},
	{	{ -0.369984f, -0.851557f, -0.369984f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.000000f, 0.618800f, -0.785600f }	},
	{	{ 0.369984f, -0.851557f, -0.369984f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.000000f, 0.618800f, -0.785600f }	},
	{	{ 0.390598f, -0.877727f, -0.390598f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.000000f, 0.618800f, -0.785600f }	},
	{	{ -0.390598f, -0.877727f, -0.390598f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.000000f, 0.618800f, -0.785600f }	},
	{	{ -0.369984f, -0.851557f, 0.369984f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.785600f, 0.618800f, 0.000000f }	},
	{	{ -0.369984f, -0.851557f, -0.369984f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.785600f, 0.618800f, 0.000000f }	},
	{	{ -0.390598f, -0.877727f, -0.390598f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.785600f, 0.618800f, 0.000000f }	},
	{	{ -0.390598f, -0.877727f, 0.390598f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.785600f, 0.618800f, 0.000000f }	},
	{	{ -0.369984f, -0.851557f, 0.369984f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.804200f, 0.594400f }	},
	{	{ 0.369984f, -0.851557f, 0.369984f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.804200f, 0.594400f }	},
	{	{ 0.273744f, -0.780422f, 0.273744f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.804200f, 0.594400f }	},
	{	{ -0.273744f, -0.780422f, 0.273744f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.804200f, 0.594400f }	},
	{	{ 1.000000f, -0.220983f, 0.084656f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, 1.000000f }	},
	{	{ 1.000000f, -0.033099f, 0.084656f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, 1.000000f }	},
	{	{ -1.000000f, -0.033099f, 0.084656f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, 1.000000f }	},
	{	{ -1.000000f, -0.220983f, 0.084656f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, 1.000000f }	},
	{	{ 1.000000f, -0.220983f, -0.084656f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 1.000000f, 0.000000f, -0.000000f }	},
	{	{ 1.000000f, -0.033099f, -0.084656f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 1.000000f, 0.000000f, -0.000000f }	},
	{	{ 1.000000f, -0.033099f, 0.084656f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 1.000000f, 0.000000f, -0.000000f }	},
	{	{ 1.000000f, -0.220983f, 0.084656f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 1.000000f, 0.000000f, -0.000000f }	},
	{	{ -1.000000f, -0.220983f, -0.084656f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, -1.000000f }	},
	{	{ -1.000000f, -0.033099f, -0.084656f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, -1.000000f }	},
	{	{ 1.000000f, -0.033099f, -0.084656f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, -1.000000f }	},
	{	{ 1.000000f, -0.220983f, -0.084656f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, -1.000000f }	},
	{	{ -1.000000f, -0.220983f, 0.084656f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -1.000000f, 0.000000f, -0.000000f }	},
	{	{ -1.000000f, -0.033099f, 0.084656f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -1.000000f, 0.000000f, -0.000000f }	},
	{	{ -1.000000f, -0.033099f, -0.084656f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -1.000000f, 0.000000f, -0.000000f }	},
	{	{ -1.000000f, -0.220983f, -0.084656f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -1.000000f, 0.000000f, -0.000000f }	},
	{	{ -1.000000f, -0.220983f, 0.084656f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -1.000000f, 0.000000f }	},
	{	{ -1.000000f, -0.220983f, -0.084656f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -1.000000f, 0.000000f }	},
	{	{ 1.000000f, -0.220983f, -0.084656f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -1.000000f, 0.000000f }	},
	{	{ 1.000000f, -0.220983f, 0.084656f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -1.000000f, 0.000000f }	},
	{	{ -1.000000f, -0.033099f, -0.084656f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ -1.000000f, -0.033099f, 0.084656f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ 1.000000f, -0.033099f, 0.084656f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ 1.000000f, -0.033099f, -0.084656f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ -0.166718f, 0.294840f, -0.166718f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ -0.166718f, 0.294840f, 0.166718f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ 0.166718f, 0.294840f, 0.166718f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ 0.166718f, 0.294840f, -0.166718f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, -0.000000f }	},
	{	{ -0.206189f, -0.273684f, -0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, -1.000000f }	},
	{	{ -0.206189f, 0.231141f, -0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, -1.000000f }	},
	{	{ 0.206188f, 0.231141f, -0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, -1.000000f }	},
	{	{ 0.206188f, -0.273684f, -0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, -1.000000f }	},
	{	{ 0.206189f, -0.273684f, 0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, 1.000000f }	},
	{	{ 0.206189f, 0.231141f, 0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, 1.000000f }	},
	{	{ -0.206188f, 0.231141f, 0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, 1.000000f }	},
	{	{ -0.206188f, -0.273684f, 0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, 1.000000f }	},
	{	{ 0.206188f, -0.273684f, -0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 1.000000f, 0.000000f, -0.000000f }	},
	{	{ 0.206188f, 0.231141f, -0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 1.000000f, 0.000000f, -0.000000f }	},
	{	{ 0.206189f, 0.231141f, 0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 1.000000f, 0.000000f, -0.000000f }	},
	{	{ 0.206189f, -0.273684f, 0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 1.000000f, 0.000000f, -0.000000f }	},
	{	{ -0.206188f, 0.231141f, 0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.526700f, 0.850000f }	},
	{	{ 0.206189f, 0.231141f, 0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.526700f, 0.850000f }	},
	{	{ 0.166718f, 0.294840f, 0.166718f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.526700f, 0.850000f }	},
	{	{ -0.166718f, 0.294840f, 0.166718f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.526700f, 0.850000f }	},
	{	{ 0.206189f, 0.231141f, 0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.850000f, 0.526700f, -0.000000f }	},
	{	{ 0.206188f, 0.231141f, -0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.850000f, 0.526700f, -0.000000f }	},
	{	{ 0.166718f, 0.294840f, -0.166718f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.850000f, 0.526700f, -0.000000f }	},
	{	{ 0.166718f, 0.294840f, 0.166718f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.850000f, 0.526700f, -0.000000f }	},
	{	{ 0.206188f, 0.231141f, -0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.526700f, -0.850000f }	},
	{	{ -0.206189f, 0.231141f, -0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.526700f, -0.850000f }	},
	{	{ -0.166718f, 0.294840f, -0.166718f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.526700f, -0.850000f }	},
	{	{ 0.166718f, 0.294840f, -0.166718f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.526700f, -0.850000f }	},
	{	{ -0.206189f, 0.231141f, -0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.850000f, 0.526700f, 0.000000f }	},
	{	{ -0.206188f, 0.231141f, 0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.850000f, 0.526700f, 0.000000f }	},
	{	{ -0.166718f, 0.294840f, 0.166718f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.850000f, 0.526700f, 0.000000f }	},
	{	{ -0.166718f, 0.294840f, -0.166718f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.850000f, 0.526700f, 0.000000f }	},
	{	{ -0.206188f, -0.273684f, 0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -1.000000f, 0.000000f, -0.000000f }	},
	{	{ -0.206188f, 0.231141f, 0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -1.000000f, 0.000000f, -0.000000f }	},
	{	{ -0.206189f, 0.231141f, -0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -1.000000f, 0.000000f, -0.000000f }	},
	{	{ -0.206189f, -0.273684f, -0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -1.000000f, 0.000000f, -0.000000f }	},
	{	{ -0.206188f, -0.426092f, 0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -1.000000f, 0.000000f, -0.000000f }	},
	{	{ -0.206188f, -0.303970f, 0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -1.000000f, 0.000000f, -0.000000f }	},
	{	{ -0.206189f, -0.303970f, -0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -1.000000f, 0.000000f, -0.000000f }	},
	{	{ -0.206189f, -0.426092f, -0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -1.000000f, 0.000000f, -0.000000f }	},
	{	{ 0.178376f, -0.277789f, 0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.146000f, -0.989300f, 0.000000f }	},
	{	{ 0.178376f, -0.277789f, -0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.146000f, -0.989300f, 0.000000f }	},
	{	{ 0.206188f, -0.273684f, -0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.146000f, -0.989300f, 0.000000f }	},
	{	{ 0.206189f, -0.273684f, 0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.146000f, -0.989300f, 0.000000f }	},
	{	{ 0.206188f, -0.426092f, -0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 1.000000f, 0.000000f, -0.000000f }	},
	{	{ 0.206188f, -0.303970f, -0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 1.000000f, 0.000000f, -0.000000f }	},
	{	{ 0.206189f, -0.303970f, 0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 1.000000f, 0.000000f, -0.000000f }	},
	{	{ 0.206189f, -0.426092f, 0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 1.000000f, 0.000000f, -0.000000f }	},
	{	{ -0.178376f, -0.277789f, 0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.989300f, 0.146000f }	},
	{	{ 0.178376f, -0.277789f, 0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.989300f, 0.146000f }	},
	{	{ 0.206189f, -0.273684f, 0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.989300f, 0.146000f }	},
	{	{ -0.206188f, -0.273684f, 0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, -0.989300f, 0.146000f }	},
	{	{ 0.206189f, -0.426092f, 0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, 1.000000f }	},
	{	{ 0.206189f, -0.303970f, 0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, 1.000000f }	},
	{	{ -0.206188f, -0.303970f, 0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, 1.000000f }	},
	{	{ -0.206188f, -0.426092f, 0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, 1.000000f }	},
	{	{ 0.178376f, -0.277789f, -0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.000000f, -0.989300f, -0.146000f }	},
	{	{ -0.178376f, -0.277789f, -0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.000000f, -0.989300f, -0.146000f }	},
	{	{ -0.206189f, -0.273684f, -0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.000000f, -0.989300f, -0.146000f }	},
	{	{ 0.206188f, -0.273684f, -0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.000000f, -0.989300f, -0.146000f }	},
	{	{ -0.206189f, -0.426092f, -0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, -1.000000f }	},
	{	{ -0.206189f, -0.303970f, -0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, -1.000000f }	},
	{	{ 0.206188f, -0.303970f, -0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, -1.000000f }	},
	{	{ 0.206188f, -0.426092f, -0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, -1.000000f }	},
	{	{ -0.178376f, -0.303989f, 0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000700f, 1.000000f, 0.000000f }	},
	{	{ -0.178376f, -0.303989f, -0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000700f, 1.000000f, 0.000000f }	},
	{	{ -0.206189f, -0.303970f, -0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000700f, 1.000000f, 0.000000f }	},
	{	{ -0.206188f, -0.303970f, 0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000700f, 1.000000f, 0.000000f }	},
	{	{ -0.178376f, -0.303989f, 0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -1.000000f, 0.000000f, -0.000000f }	},
	{	{ -0.178376f, -0.277789f, 0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -1.000000f, 0.000000f, -0.000000f }	},
	{	{ -0.178376f, -0.277789f, -0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -1.000000f, 0.000000f, -0.000000f }	},
	{	{ -0.178376f, -0.303989f, -0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -1.000000f, 0.000000f, -0.000000f }	},
	{	{ 0.178376f, -0.303989f, -0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 1.000000f, 0.000000f, -0.000000f }	},
	{	{ 0.178376f, -0.277789f, -0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 1.000000f, 0.000000f, -0.000000f }	},
	{	{ 0.178376f, -0.277789f, 0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 1.000000f, 0.000000f, -0.000000f }	},
	{	{ 0.178376f, -0.303989f, 0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 1.000000f, 0.000000f, -0.000000f }	},
	{	{ 0.178376f, -0.303989f, 0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, 1.000000f }	},
	{	{ 0.178376f, -0.277789f, 0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, 1.000000f }	},
	{	{ -0.178376f, -0.277789f, 0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, 1.000000f }	},
	{	{ -0.178376f, -0.303989f, 0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, 1.000000f }	},
	{	{ -0.178376f, -0.303989f, -0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, -1.000000f }	},
	{	{ -0.178376f, -0.277789f, -0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, -1.000000f }	},
	{	{ 0.178376f, -0.277789f, -0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, -1.000000f }	},
	{	{ 0.178376f, -0.303989f, -0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 0.000000f, -1.000000f }	},
	{	{ -0.178376f, -0.277789f, -0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.146000f, -0.989300f, -0.000000f }	},
	{	{ -0.178376f, -0.277789f, 0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.146000f, -0.989300f, -0.000000f }	},
	{	{ -0.206188f, -0.273684f, 0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.146000f, -0.989300f, -0.000000f }	},
	{	{ -0.206189f, -0.273684f, -0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.146000f, -0.989300f, -0.000000f }	},
	{	{ -0.178376f, -0.303989f, -0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, 0.000700f }	},
	{	{ 0.178376f, -0.303989f, -0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, 0.000700f }	},
	{	{ 0.206188f, -0.303970f, -0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, 0.000700f }	},
	{	{ -0.206189f, -0.303970f, -0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ 0.000000f, 1.000000f, 0.000700f }	},
	{	{ 0.178376f, -0.303989f, 0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.000000f, 1.000000f, -0.000700f }	},
	{	{ -0.178376f, -0.303989f, 0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.000000f, 1.000000f, -0.000700f }	},
	{	{ -0.206188f, -0.303970f, 0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.000000f, 1.000000f, -0.000700f }	},
	{	{ 0.206189f, -0.303970f, 0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.000000f, 1.000000f, -0.000700f }	},
	{	{ 0.178376f, -0.303989f, -0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.000700f, 1.000000f, -0.000000f }	},
	{	{ 0.178376f, -0.303989f, 0.178376f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.000700f, 1.000000f, -0.000000f }	},
	{	{ 0.206189f, -0.303970f, 0.206188f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.000700f, 1.000000f, -0.000000f }	},
	{	{ 0.206188f, -0.303970f, -0.206189f },	{ 0.000000f, 0.000000f, 0.000000f },	{ -0.000700f, 1.000000f, -0.000000f }	},
};
#define _Fence_Straight_Modular_vertices_
#endif
#ifndef _Fence_Straight_Modular_indices_
// Index Data follows: Sequential by mesh, Every Three Indices == One Triangle.
const unsigned int Fence_Straight_Modular_indices[312] =
{
	 0, 1, 2,
	 3, 0, 2,
	 4, 5, 6,
	 7, 4, 6,
	 8, 0, 3,
	 9, 8, 3,
	 1, 10, 11,
	 2, 1, 11,
	 10, 8, 9,
	 11, 10, 9,
	 12, 13, 14,
	 15, 12, 14,
	 16, 17, 18,
	 19, 16, 18,
	 20, 21, 22,
	 23, 20, 22,
	 24, 25, 26,
	 27, 24, 26,
	 28, 29, 30,
	 31, 28, 30,
	 32, 33, 34,
	 35, 32, 34,
	 36, 37, 38,
	 39, 36, 38,
	 40, 41, 42,
	 43, 40, 42,
	 44, 45, 46,
	 47, 44, 46,
	 48, 49, 50,
	 51, 48, 50,
	 52, 53, 54,
	 55, 52, 54,
	 56, 57, 58,
	 59, 56, 58,
	 60, 61, 62,
	 63, 60, 62,
	 64, 65, 66,
	 67, 64, 66,
	 68, 69, 70,
	 71, 68, 70,
	 72, 73, 74,
	 75, 72, 74,
	 76, 77, 78,
	 79, 76, 78,
	 80, 81, 82,
	 83, 80, 82,
	 84, 85, 86,
	 87, 84, 86,
	 88, 89, 90,
	 91, 88, 90,
	 92, 93, 94,
	 95, 92, 94,
	 96, 97, 98,
	 99, 96, 98,
	 100, 101, 102,
	 103, 100, 102,
	 104, 105, 106,
	 107, 104, 106,
	 108, 109, 110,
	 111, 108, 110,
	 112, 113, 114,
	 115, 112, 114,
	 116, 117, 118,
	 119, 116, 118,
	 120, 121, 122,
	 123, 120, 122,
	 124, 125, 126,
	 127, 124, 126,
	 128, 129, 130,
	 131, 128, 130,
	 132, 133, 134,
	 135, 132, 134,
	 136, 137, 138,
	 139, 136, 138,
	 140, 141, 142,
	 143, 140, 142,
	 144, 145, 146,
	 147, 144, 146,
	 148, 149, 150,
	 151, 148, 150,
	 152, 153, 154,
	 155, 152, 154,
	 156, 157, 158,
	 159, 156, 158,
	 160, 161, 162,
	 163, 160, 162,
	 164, 165, 166,
	 167, 164, 166,
	 168, 169, 170,
	 171, 168, 170,
	 172, 173, 174,
	 175, 172, 174,
	 176, 177, 178,
	 179, 176, 178,
	 180, 181, 182,
	 183, 180, 182,
	 184, 185, 186,
	 187, 184, 186,
	 188, 189, 190,
	 191, 188, 190,
	 192, 193, 194,
	 195, 192, 194,
	 196, 197, 198,
	 199, 196, 198,
};
#define _Fence_Straight_Modular_indices_
#endif
// Part of an OBJ_MATERIAL, the struct is 16 byte aligned so it is GPU register friendly.
#ifndef __OBJ_ATTRIBUTES__
typedef struct _OBJ_ATTRIBUTES_
{
	OBJ_VEC3    Kd; // diffuse reflectivity
	float	    d; // dissolve (transparency) 
	OBJ_VEC3    Ks; // specular reflectivity
	float       Ns; // specular exponent
	OBJ_VEC3    Ka; // ambient reflectivity
	float       sharpness; // local reflection map sharpness
	OBJ_VEC3    Tf; // transmission filter
	float       Ni; // optical density (index of refraction)
	OBJ_VEC3    Ke; // emissive reflectivity
	unsigned    illum; // illumination model
}OBJ_ATTRIBUTES;
#define __OBJ_ATTRIBUTES__
#endif
// This structure also has been made GPU register friendly so it can be transfered directly if desired.
// Note: Total struct size will vary depenedening on CPU processor architecture (string pointers).
#ifndef __OBJ_MATERIAL__
typedef struct _OBJ_MATERIAL_
{
	// The following items are typically used in a pixel/fragment shader, they are packed for GPU registers.
	OBJ_ATTRIBUTES attrib; // Surface shading characteristics & illumination model
	// All items below this line are not needed on the GPU and are generally only used during load time.
	const char* name; // the name of this material
	// If the model's materials contain any specific texture data it will be located below.
	const char* map_Kd; // diffuse texture
	const char* map_Ks; // specular texture
	const char* map_Ka; // ambient texture
	const char* map_Ke; // emissive texture
	const char* map_Ns; // specular exponent texture
	const char* map_d; // transparency texture
	const char* disp; // roughness map (displacement)
	const char* decal; // decal texture (lerps texture & material colors)
	const char* bump; // normal/bumpmap texture
	void* padding[2]; // 16 byte alignment on 32bit or 64bit
}OBJ_MATERIAL;
#define __OBJ_MATERIAL__
#endif
#ifndef _Fence_Straight_Modular_materials_
// Material Data follows: pulled from a .mtl file of the same name if present.
const OBJ_MATERIAL Fence_Straight_Modular_materials[2] =
{
	{
		{{ 0.036393f, 0.033784f, 0.048574f },
		1.000000f,
		{ 0.500000f, 0.500000f, 0.500000f },
		96.078430f,
		{ 1.000000f, 1.000000f, 1.000000f },
		60.000000f,
		{ 1.000000f, 1.000000f, 1.000000f },
		1.000000f,
		{ 0.000000f, 0.000000f, 0.000000f },
		2},
		"DarkGrey_Floor",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
	},
	{
		{{ 0.059253f, 0.057173f, 0.081973f },
		1.000000f,
		{ 0.500000f, 0.500000f, 0.500000f },
		96.078430f,
		{ 1.000000f, 1.000000f, 1.000000f },
		60.000000f,
		{ 1.000000f, 1.000000f, 1.000000f },
		1.000000f,
		{ 0.000000f, 0.000000f, 0.000000f },
		2},
		"Grey_Floor",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
	},
};
#define _Fence_Straight_Modular_materials_
#endif
/************************************************/
/*  This section contains the model's structure */
/************************************************/
#ifndef _Fence_Straight_Modular_batches_
// Use this conveinence array to batch render all geometry by like material.
// Each entry corresponds to the same entry in the materials array above.
// The two numbers provided are the IndexCount and the IndexOffset into the indices array.
// If you need more fine grained control(ex: for transformations) use the OBJ_MESH data below.
const unsigned int Fence_Straight_Modular_batches[2][2] =
{
	{ 162, 0 },
	{ 150, 162 },
};
#define _Fence_Straight_Modular_batches_
#endif
#ifndef __OBJ_MESH__
typedef struct _OBJ_MESH_
{
	const char* name;
	unsigned    indexCount;
	unsigned    indexOffset;
	unsigned    materialIndex;
}OBJ_MESH;
#define __OBJ_MESH__
#endif
#ifndef _Fence_Straight_Modular_meshes_
// Mesh Data follows: Meshes are .obj groups sorted & split by material.
// Meshes are provided in sequential order, sorted by material first and name second.
const OBJ_MESH Fence_Straight_Modular_meshes[2] =
{
	{
		"default",
		162,
		0,
		0,
	},
	{
		"default",
		150,
		162,
		1,
	},
};
#define _Fence_Straight_Modular_meshes_
#endif
