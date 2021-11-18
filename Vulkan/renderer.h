// minimalistic code to draw a single triangle, this is not part of the API.
// TODO: Part 1b
#include "FSLogo.h"
#include "h2bParser.h"
#include "shaderc/shaderc.h" // needed for compiling shaders at runtime
#define MAX_SUBMESH_PER_DRAW 1024
#define PLAYMUSIC false
#ifdef _WIN32 // must use MT platform DLL libraries on windows
	#pragma comment(lib, "shaderc_combined.lib") 
#endif
// Simple Vertex Shader
const char* vertexShaderSource = R"(
#pragma pack_matrix(row_major)
// TODO: 2i
// an ultra simple hlsl vertex shader
// TODO: Part 2b
struct ATTRIBUTES
{
    float3 Kd; // diffuse reflectivity
    float d; // dissolve (transparency) 
    float3 Ks; // specular reflectivity
    float Ns; // specular exponent
    float3 Ka; // ambient reflectivity
    float sharpness; // local reflection map sharpness
    float3 Tf; // transmission filter
    float Ni; // optical density (index of refraction)
    float3 Ke; // emissive reflectivity
    uint illum; // illumination model
};
struct SHADER_MODEL_DATA
{
    float4 sunDirection, sunColor;
    matrix ViewMatrix, ProjectionMatrix;
    matrix matricies[1024];
    ATTRIBUTES materials[1024];
	float4 sunAmbient, camPos;
};
StructuredBuffer<SHADER_MODEL_DATA> SceneData;
// TODO: Part 4g
// TODO: Part 2i
// TODO: Part 3e
[[vk::push_constant]]
cbuffer MESH_INDEX {
	uint mesh_ID;
	uint wm_ID;
};
// TODO: Part 4a
// TODO: Part 1f
struct OBJ_VERT
{
    float3 pos : POSITION;
    float3 uvw : TEXCOORD0;
    float3 nrm : NORMAL;
};
struct OutputStruct
{
	float4 posH : SV_POSITION;
	float3 nrmW : NORMAL;
	float3 posW : WORLD;
	float3 uvw : TEXTCOORD0;
};
// TODO: Part 4b
OutputStruct main(OBJ_VERT inputObj, uint instanceId : SV_INSTANCEID)
{
    // TODO: Part 1h
    //inputObj.pos[2] += 0.75f;
    //inputObj.pos[1] -= 0.75f;
	// TODO: Part 2i
	OutputStruct tempStruct;
	float4 pos = float4(inputObj.pos, 1);
	tempStruct.posH = mul(pos, SceneData[0].matricies[instanceId]);
	tempStruct.posW = tempStruct.posH;
	tempStruct.posH = mul(tempStruct.posH, SceneData[0].ViewMatrix);
	tempStruct.posH = mul(tempStruct.posH, SceneData[0].ProjectionMatrix);
	tempStruct.uvw = inputObj.uvw;
		// TODO: Part 4e
	// TODO: Part 4b
	tempStruct.nrmW = normalize(inputObj.nrm);
	tempStruct.nrmW = mul(tempStruct.nrmW, SceneData[0].matricies[instanceId]);
	//tempStruct.posW = mul(inputObj.pos, SceneData[0].matricies[instanceId]);
		// TODO: Part 4e
    return tempStruct;
}
)";
// Simple Pixel Shader
const char* pixelShaderSource = R"(
struct ATTRIBUTES
{
    float3 Kd; // diffuse reflectivity
    float d; // dissolve (transparency) 
    float3 Ks; // specular reflectivity
    float Ns; // specular exponent
    float3 Ka; // ambient reflectivity
    float sharpness; // local reflection map sharpness
    float3 Tf; // transmission filter
    float Ni; // optical density (index of refraction)
    float3 Ke; // emissive reflectivity
    uint illum; // illumination model
};
struct SHADER_MODEL_DATA
{
    float4 sunDirection, sunColor;
    matrix ViewMatrix, ProjectionMatrix;
    matrix matricies[1024];
    ATTRIBUTES materials[1024];
    float4 sunAmbient, camPos;
};
StructuredBuffer<SHADER_MODEL_DATA> SceneData;
// TODO: Part 4g
// TODO: Part 2i
// TODO: Part 3e
[[vk::push_constant]]
cbuffer MESH_INDEX
{
    uint mesh_ID;
    uint wm_ID;
};
// TODO: Part 4a
// TODO: Part 1f
struct OBJ_VERT
{
    float3 pos : POSITION;
    float3 uvw : TEXCOORD0;
    float3 nrm : NORMAL;
};
struct InputStruct
{
    float4 posH : SV_POSITION;
    float3 nrmW : NORMAL;
    float3 posW : WORLD;
    float3 uvw : TEXTCOORD0;
};

// an ultra simple hlsl pixel shader
// TODO: Part 4b


float4 main(InputStruct inputObj) : SV_TARGET
{

	inputObj.nrmW = normalize(inputObj.nrmW);
	float3 lightDir = normalize(SceneData[0].sunDirection.xyz);

	float4 final;
	final = saturate(dot(-lightDir, inputObj.nrmW)) * SceneData[0].sunColor;
	final += SceneData[0].sunAmbient;
	final *= float4(SceneData[0].materials[mesh_ID].Kd, 0.0f);
	
	float3 viewDir = normalize(SceneData[0].camPos.xyz - inputObj.posW);
	float3 reflected = reflect(lightDir, inputObj.nrmW);
	float intensity = pow(saturate(dot(viewDir, reflected)), SceneData[0].materials[mesh_ID].Ns);
	final += float4(SceneData[0].sunColor.xyz * SceneData[0].materials[mesh_ID].Ks * intensity, 0.0f);

	return final;
}
)";
	////return float4(inputObj.nrmW, 1);
 //   float lightratio = saturate(dot(-normalize(SceneData[0].sunDirection), normalize(inputObj.nrmW)));
 //   float3 result = (saturate(lightratio + SceneData[0].sunAmbient) * SceneData[0].sunColor) * SceneData[0].materials[mesh_ID].Kd;
	////result = lightratio * SceneData[0].sunColor;
	//float4 specular = CalculateSpecular(SceneData[0].camPos.xyz, SceneData[0].sunDirection, inputObj.posW, inputObj.nrmW, SceneData[0].materials[mesh_ID]);


 //   //float3 viewDir = normalize(SceneData[0].camPos.xyz - inputObj.posW);
 //   //float3 halfVector = normalize(-SceneData[0].sunDirection + viewDir);
 //   //float intensity = pow(saturate(dot(normalize(inputObj.nrmW), halfVector)), SceneData[0].materials[mesh_ID].Ns);
 //   //float3 reflectedLight = /*SceneData[0].sunColor **/ SceneData[0].materials[mesh_ID].Ks * intensity;

	//return float4(SceneData[0].sunAmbient + result, 1) * float4(SceneData[0].materials[mesh_ID].Kd, 1) + specular;
 //   //return float4(SceneData[0].sunAmbient + result, 1) * float4(SceneData[0].materials[mesh_ID].Kd, 1) + float4(reflectedLight, 0); // TODO: Part 1a

struct SHADER_MODEL_DATA {
	GW::MATH::GVECTORF sunDirection, sunColor;
	GW::MATH::GMATRIXF ViewMatrix, ProjectionMatrix;
	GW::MATH::GMATRIXF matricies[MAX_SUBMESH_PER_DRAW];
	H2B::ATTRIBUTES materials[MAX_SUBMESH_PER_DRAW];
	GW::MATH::GVECTORF sunAmbient, camPos;
};
struct Mesh_Struct {
	std::vector<GW::MATH::GMATRIXF> WorldMatrices;
	std::string filename;
	std::string filepath;
	H2B::Parser h2bParser;
};
struct MESH_INDEX {
	unsigned mesh_ID;
	unsigned wm_ID;
};
// Creation, Rendering & Cleanup
class Renderer
{
	// TODO: Part 2b
	SHADER_MODEL_DATA ShaderModelData;
	
	// proxy handles
	GW::SYSTEM::GWindow win;
	GW::GRAPHICS::GVulkanSurface vlk;
	GW::CORE::GEventReceiver shutdown;

	GW::AUDIO::GMusic GMusic;
	GW::AUDIO::GSound GSound;
	GW::AUDIO::GAudio GAudio;
	std::vector<const char*> Songs;
	GW::INPUT::GInput Ginput;
	std::chrono::steady_clock::time_point now;

	std::map<std::string, Mesh_Struct> MeshMap;
	std::vector<H2B::VERTEX> Vertexes;
	std::vector<unsigned> Indexes;
	std::vector<unsigned> VertexOffsets;
	std::vector<unsigned> IndexOffsets;
	std::vector<unsigned> MatrixOffset;
	std::vector<unsigned> MaterialOffset;
	int VertexSize;
	int IndexSize;
	int SongChoice = 0;
	int CurrentScene = 0;
	// what we need at a minimum to draw a triangle
	VkDevice device = nullptr;
	VkBuffer vertexHandle = nullptr;
	VkDeviceMemory vertexData = nullptr;
	// TODO: Part 1g
	VkBuffer indiciesBuffer = nullptr;
	VkDeviceMemory indiciesData = nullptr;
	// TODO: Part 2c
	std::vector<VkBuffer> VectorVkBuffer;
	std::vector<VkDeviceMemory> VectorDeviceMemory;
	VkShaderModule vertexShader = nullptr;
	VkShaderModule pixelShader = nullptr;
	// pipeline settings for drawing (also required).
	VkPipeline pipeline = nullptr;
	VkPipelineLayout pipelineLayout = nullptr;
	// TODO: Part 2e
	VkDescriptorSetLayout descriptorSetLayout = nullptr;
	// TODO: Part 2f
	VkDescriptorPool descriptorPool = nullptr;
	// TODO: Part 2g
	std::vector<VkDescriptorSet> descriptorSet;
		// TODO: Part 4f
		
	// TODO: Part 2a
	GW::MATH::GMatrix MatrixProxy;
	GW::MATH::GVector VectorProxy;
	GW::MATH::GMATRIXF WorldMatrix;
	GW::MATH::GMATRIXF SecondWorldMatrix;
	GW::MATH::GMATRIXF ViewMatrix;
	GW::MATH::GMATRIXF ProjectionMatrix;
	GW::MATH::GVECTORF LightDirectionVector;
	GW::MATH::GVECTORF LightColorVector;
	// TODO: Part 2b
	// TODO: Part 4g
public:

	bool ParseFile(const char* file) {
		std::string line;
		std::ifstream myfile(file);
		if (!myfile.is_open()) {
			std::cout << "Unable to open file";
			return false;
		}
		while (std::getline(myfile, line)) {
			if (line == "MESH") {
				std::getline(myfile, line);
				std::string delimiter = ".";
				std::string MeshName = line.substr(0, line.find(delimiter));
				//std::cout << MeshName << "\n";
				GW::MATH::GMATRIXF Matrix;
				int i = 0;
				for (int y = 0; y < 4; y++) {
					std::getline(myfile, line);
					delimiter = ")";
					size_t pos = 0;
					std::string token;
					line = line.substr(13, (line.find(delimiter) - 13));
					delimiter = ",";
					for (int o = 0; o < 4; o++) {
						pos = line.find(delimiter);
						token = line.substr(0, pos);
						Matrix.data[i] = std::stof(token);
						line.erase(0, pos + delimiter.length());
						i++;
					}
				}
				if (MeshMap.find(MeshName) == MeshMap.end()) {
					std::string filePath = "../../";
					filePath.append(MeshName);
					filePath.append(".h2b");
					Mesh_Struct tStruct = { };
					tStruct = { {Matrix}, MeshName, filePath };
					MeshMap.insert(std::pair<std::string, Mesh_Struct>(MeshName, tStruct));
					MeshMap[MeshName].h2bParser.Parse(filePath.c_str());
				} else
					MeshMap[MeshName].WorldMatrices.push_back(Matrix);
			}
		}
		int Vcounter = 0;
		int Icounter = 0;
		int mCount = 0;
		int matCount = 0;
		VertexOffsets.push_back(0);
		IndexOffsets.push_back(0);
		MatrixOffset.push_back(0);
		MaterialOffset.push_back(0);
		for (std::map<std::string, Mesh_Struct>::iterator it = MeshMap.begin(); it != MeshMap.end(); ++it) {
			Vertexes.insert(Vertexes.end(), it->second.h2bParser.vertices.begin(), it->second.h2bParser.vertices.end());
			Indexes.insert(Indexes.end(), it->second.h2bParser.indices.begin(), it->second.h2bParser.indices.end());
			Vcounter += it->second.h2bParser.vertices.size();
			VertexOffsets.push_back(Vcounter);
			Icounter += it->second.h2bParser.indices.size();
			IndexOffsets.push_back(Icounter);
			mCount += it->second.WorldMatrices.size();
			MatrixOffset.push_back(mCount);
			matCount += it->second.h2bParser.materialCount;
			MaterialOffset.push_back(matCount);
		}
		//sizeof(std::vector<int>) + (sizeof(int) * MyVector.size())
		VertexSize = sizeof(std::vector<H2B::VECTOR>) + (sizeof(H2B::VECTOR) * Vertexes.size());
		IndexSize = sizeof(Indexes) * Indexes.size();
		myfile.close();
		return true;
	}
	Renderer(GW::SYSTEM::GWindow _win, GW::GRAPHICS::GVulkanSurface _vlk)
	{
		win = _win;
		vlk = _vlk;
		MatrixProxy.Create();
		VectorProxy.Create();
		Ginput.Create(win);
		now = std::chrono::steady_clock::now();
		unsigned int width, height;
		win.GetClientWidth(width);
		win.GetClientHeight(height);
		if (!ParseFile("../GameLevel.txt"))
			exit(69);
		//ShowCursor(false);
		// TODO: Part 2a
		if (PLAYMUSIC) {
			Songs.push_back("../SayGoodbye.wav");
			Songs.push_back("../SadLittleMan.wav");
			Songs.push_back("../AllStar.wav");
			GAudio.Create();
			GMusic.Create(Songs[SongChoice], GAudio);
			GMusic.Play();
		}

		MatrixProxy.IdentityF(WorldMatrix);
		GW::MATH::GVECTORF _eye = { 0.75f, 0.25f, -1.5f, 0 };
		GW::MATH::GVECTORF _at = { 0.15f, 0.75f, 0, 0 };
		GW::MATH::GVECTORF _up = { 0, 1, 0, 0 };
		MatrixProxy.LookAtLHF(_eye, _at, _up, ViewMatrix);
		float ar = 0;
		vlk.GetAspectRatio(ar);
		MatrixProxy.ProjectionVulkanLHF(1.13446f, ar, 0.1f, 100.0f, ProjectionMatrix);
		LightDirectionVector = { -1.0f, -1.0f, 1.0f, 0.0f };
		//VectorProxy.NormalizeF(LightDirectionVector, LightDirectionVector);
		LightColorVector = { 0.9f, 0.9f, 1.0f, 1.0f };
		// TODO: Part 2b
		ShaderModelData.matricies[0] = WorldMatrix;
		ShaderModelData.sunDirection = LightDirectionVector;
		ShaderModelData.sunColor = LightColorVector;
		ShaderModelData.ViewMatrix = ViewMatrix;
		ShaderModelData.ProjectionMatrix = ProjectionMatrix;
		int externalI = 0;
		int externalZ = 0;
		for (std::map<std::string, Mesh_Struct>::iterator it = MeshMap.begin(); it != MeshMap.end(); ++it) {
			for (int y = 0; y < it->second.h2bParser.materialCount; y++, externalI++)
				ShaderModelData.materials[externalI] = it->second.h2bParser.materials[y].attrib;
			for (int y = 0; y < it->second.WorldMatrices.size(); y++, externalZ++)
				ShaderModelData.matricies[externalZ] = it->second.WorldMatrices[y];
		}
			
		
		// TODO: Part 4g
		ShaderModelData.sunAmbient = { 0.25f, 0.25f, 0.35f, 1 };
		ShaderModelData.camPos = { 0.75f, 0.25f, -1.5f, 1 };
		// TODO: part 3b

		/***************** GEOMETRY INTIALIZATION ******************/
		// Grab the device & physical device so we can allocate some stuff
		VkPhysicalDevice physicalDevice = nullptr;
		vlk.GetDevice((void**)&device);
		vlk.GetPhysicalDevice((void**)&physicalDevice);

		// TODO: Part 1c
		// Create Vertex Buffer
		// Transfer triangle data to the vertex buffer. (staging would be prefered here)
		H2B::VERTEX* tempVec = &Vertexes[0];
		GvkHelper::create_buffer(physicalDevice, device, sizeof(*tempVec) * Vertexes.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &vertexHandle, &vertexData);
		GvkHelper::write_to_buffer(device, vertexData, &*tempVec, sizeof(*tempVec) * Vertexes.size());
		// TODO: Part 1g
		unsigned* tempInd = &Indexes[0];
		GvkHelper::create_buffer(physicalDevice, device, sizeof(*tempInd) * Indexes.size(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &indiciesBuffer, &indiciesData);
		GvkHelper::write_to_buffer(device, indiciesData, &*tempInd, sizeof(*tempInd) * Indexes.size());

		// TODO: Part 2d
		unsigned count = 0;
		vlk.GetSwapchainImageCount(count);
		VectorDeviceMemory.resize(count);
		VectorVkBuffer.resize(count);
		descriptorSet.resize(count);
		for (int i = 0; i < count; i++) {
			GvkHelper::create_buffer(physicalDevice, device, sizeof(ShaderModelData),
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
				VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &VectorVkBuffer[i], &VectorDeviceMemory[i]);
			GvkHelper::write_to_buffer(device, VectorDeviceMemory[i], &ShaderModelData, sizeof(ShaderModelData));
		}
		/***************** SHADER INTIALIZATION ******************/
		// Intialize runtime shader compiler HLSL -> SPIRV
		shaderc_compiler_t compiler = shaderc_compiler_initialize();
		shaderc_compile_options_t options = shaderc_compile_options_initialize();
		shaderc_compile_options_set_source_language(options, shaderc_source_language_hlsl);
		shaderc_compile_options_set_invert_y(options, false); // TODO: Part 2i
#ifndef NDEBUG
		shaderc_compile_options_set_generate_debug_info(options);
#endif
		// Create Vertex Shader
		shaderc_compilation_result_t result = shaderc_compile_into_spv( // compile
			compiler, vertexShaderSource, strlen(vertexShaderSource),
			shaderc_vertex_shader, "main.vert", "main", options);
		if (shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success) // errors?
			std::cout << "Vertex Shader Errors: " << shaderc_result_get_error_message(result) << std::endl;
		GvkHelper::create_shader_module(device, shaderc_result_get_length(result), // load into Vulkan
			(char*)shaderc_result_get_bytes(result), &vertexShader);
		shaderc_result_release(result); // done
		// Create Pixel Shader
		result = shaderc_compile_into_spv( // compile
			compiler, pixelShaderSource, strlen(pixelShaderSource),
			shaderc_fragment_shader, "main.frag", "main", options);
		if (shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success) // errors?
			std::cout << "Pixel Shader Errors: " << shaderc_result_get_error_message(result) << std::endl;
		GvkHelper::create_shader_module(device, shaderc_result_get_length(result), // load into Vulkan
			(char*)shaderc_result_get_bytes(result), &pixelShader);
		shaderc_result_release(result); // done
		// Free runtime shader compiler resources
		shaderc_compile_options_release(options);
		shaderc_compiler_release(compiler);

		/***************** PIPELINE INTIALIZATION ******************/
		// Create Pipeline & Layout (Thanks Tiny!)
		VkRenderPass renderPass;
		vlk.GetRenderPass((void**)&renderPass);
		VkPipelineShaderStageCreateInfo stage_create_info[2] = {};
		// Create Stage Info for Vertex Shader
		stage_create_info[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stage_create_info[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stage_create_info[0].module = vertexShader;
		stage_create_info[0].pName = "main";
		// Create Stage Info for Fragment Shader
		stage_create_info[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stage_create_info[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stage_create_info[1].module = pixelShader;
		stage_create_info[1].pName = "main";
		// Assembly State
		VkPipelineInputAssemblyStateCreateInfo assembly_create_info = {};
		assembly_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assembly_create_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assembly_create_info.primitiveRestartEnable = false;
		// TODO: Part 1e
		// Vertex Input State
		VkVertexInputBindingDescription vertex_binding_description = {};
		vertex_binding_description.binding = 0;
		vertex_binding_description.stride = sizeof(H2B::VERTEX);
		vertex_binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		VkVertexInputAttributeDescription vertex_attribute_description[3] = {
			{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0 },
			{ 1, 0, VK_FORMAT_R32G32B32_SFLOAT, 12 },
			{ 2, 0, VK_FORMAT_R32G32B32_SFLOAT, 24 }//uv, normal, etc....
		};
		VkPipelineVertexInputStateCreateInfo input_vertex_info = {};
		input_vertex_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		input_vertex_info.vertexBindingDescriptionCount = 1;
		input_vertex_info.pVertexBindingDescriptions = &vertex_binding_description;
		input_vertex_info.vertexAttributeDescriptionCount = 3;
		input_vertex_info.pVertexAttributeDescriptions = vertex_attribute_description;
		// Viewport State (we still need to set this up even though we will overwrite the values)
		VkViewport viewport = {
            0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1
        };
        VkRect2D scissor = { {0, 0}, {width, height} };
		VkPipelineViewportStateCreateInfo viewport_create_info = {};
		viewport_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewport_create_info.viewportCount = 1;
		viewport_create_info.pViewports = &viewport;
		viewport_create_info.scissorCount = 1;
		viewport_create_info.pScissors = &scissor;
		// Rasterizer State
		VkPipelineRasterizationStateCreateInfo rasterization_create_info = {};
		rasterization_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterization_create_info.rasterizerDiscardEnable = VK_FALSE;
		rasterization_create_info.polygonMode = VK_POLYGON_MODE_FILL;
		rasterization_create_info.lineWidth = 1.0f;
		rasterization_create_info.cullMode = VK_CULL_MODE_NONE;
		rasterization_create_info.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterization_create_info.depthClampEnable = VK_FALSE;
		rasterization_create_info.depthBiasEnable = VK_FALSE;
		rasterization_create_info.depthBiasClamp = 0.0f;
		rasterization_create_info.depthBiasConstantFactor = 0.0f;
		rasterization_create_info.depthBiasSlopeFactor = 0.0f;
		// Multisampling State
		VkPipelineMultisampleStateCreateInfo multisample_create_info = {};
		multisample_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisample_create_info.sampleShadingEnable = VK_FALSE;
		multisample_create_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisample_create_info.minSampleShading = 1.0f;
		multisample_create_info.pSampleMask = VK_NULL_HANDLE;
		multisample_create_info.alphaToCoverageEnable = VK_FALSE;
		multisample_create_info.alphaToOneEnable = VK_FALSE;
		// Depth-Stencil State
		VkPipelineDepthStencilStateCreateInfo depth_stencil_create_info = {};
		depth_stencil_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depth_stencil_create_info.depthTestEnable = VK_TRUE;
		depth_stencil_create_info.depthWriteEnable = VK_TRUE;
		depth_stencil_create_info.depthCompareOp = VK_COMPARE_OP_LESS;
		depth_stencil_create_info.depthBoundsTestEnable = VK_FALSE;
		depth_stencil_create_info.minDepthBounds = 0.0f;
		depth_stencil_create_info.maxDepthBounds = 1.0f;
		depth_stencil_create_info.stencilTestEnable = VK_FALSE;
		// Color Blending Attachment & State
		VkPipelineColorBlendAttachmentState color_blend_attachment_state = {};
		color_blend_attachment_state.colorWriteMask = 0xF;
		color_blend_attachment_state.blendEnable = VK_FALSE;
		color_blend_attachment_state.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_COLOR;
		color_blend_attachment_state.dstColorBlendFactor = VK_BLEND_FACTOR_DST_COLOR;
		color_blend_attachment_state.colorBlendOp = VK_BLEND_OP_ADD;
		color_blend_attachment_state.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		color_blend_attachment_state.dstAlphaBlendFactor = VK_BLEND_FACTOR_DST_ALPHA;
		color_blend_attachment_state.alphaBlendOp = VK_BLEND_OP_ADD;
		VkPipelineColorBlendStateCreateInfo color_blend_create_info = {};
		color_blend_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		color_blend_create_info.logicOpEnable = VK_FALSE;
		color_blend_create_info.logicOp = VK_LOGIC_OP_COPY;
		color_blend_create_info.attachmentCount = 1;
		color_blend_create_info.pAttachments = &color_blend_attachment_state;
		color_blend_create_info.blendConstants[0] = 0.0f;
		color_blend_create_info.blendConstants[1] = 0.0f;
		color_blend_create_info.blendConstants[2] = 0.0f;
		color_blend_create_info.blendConstants[3] = 0.0f;
		// Dynamic State 
		VkDynamicState dynamic_state[2] = { 
			// By setting these we do not need to re-create the pipeline on Resize
			VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR
		};
		VkPipelineDynamicStateCreateInfo dynamic_create_info = {};
		dynamic_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamic_create_info.dynamicStateCount = 2;
		dynamic_create_info.pDynamicStates = dynamic_state;
		
		// TODO: Part 2e
		VkDescriptorSetLayoutBinding descriptorSetlayoutBinding = { };
		descriptorSetlayoutBinding.binding = 0;
		descriptorSetlayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorSetlayoutBinding.descriptorCount = 1;
		descriptorSetlayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = { };
		descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		descriptorSetLayoutCreateInfo.bindingCount = 1;
		descriptorSetLayoutCreateInfo.pBindings = &descriptorSetlayoutBinding;
		vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayout);
		// TODO: Part 2f
		VkDescriptorPoolSize DescriptorPoolSize = {};
		DescriptorPoolSize.descriptorCount = count;
		DescriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
		descriptorPoolCreateInfo.poolSizeCount = 1;
		descriptorPoolCreateInfo.maxSets = count;
		descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		descriptorPoolCreateInfo.pPoolSizes = &DescriptorPoolSize;
		vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr, &descriptorPool);
			// TODO: Part 4f
		// TODO: Part 2g
		VkDescriptorSetAllocateInfo descriptorsetallocateinfo = {};
		descriptorsetallocateinfo.descriptorPool = descriptorPool;
		descriptorsetallocateinfo.descriptorSetCount = 1;
		descriptorsetallocateinfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		descriptorsetallocateinfo.pSetLayouts = &descriptorSetLayout;
		for (int i = 0; i < count; i++) {
			vkAllocateDescriptorSets(device, &descriptorsetallocateinfo, &descriptorSet[i]);
		}
			// TODO: Part 4f
		// TODO: Part 2h
		for (int i = 0; i < VectorVkBuffer.size(); i++) {
			VkDescriptorBufferInfo descriptorbufferinfo = {};
			descriptorbufferinfo.buffer = VectorVkBuffer[i];
			descriptorbufferinfo.offset = 0;
			descriptorbufferinfo.range = VK_WHOLE_SIZE;
			VkWriteDescriptorSet writedescriptorset = {};
			writedescriptorset.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			writedescriptorset.dstSet = descriptorSet[i];
			writedescriptorset.dstBinding = 0;
			writedescriptorset.dstArrayElement = 0;
			writedescriptorset.descriptorCount = 1;
			writedescriptorset.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			writedescriptorset.pBufferInfo = &descriptorbufferinfo;
			writedescriptorset.pImageInfo = nullptr;
			writedescriptorset.pTexelBufferView = nullptr;
			vkUpdateDescriptorSets(device, 1, &writedescriptorset, 0, nullptr);
		}
			// TODO: Part 4f
	
		// Descriptor pipeline layout
		VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
		pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		// TODO: Part 2e
		pipeline_layout_create_info.setLayoutCount = 1;
		pipeline_layout_create_info.pSetLayouts = &descriptorSetLayout;
		// TODO: Part 3c
		VkPushConstantRange pushConstantRange = {};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
		pushConstantRange.size = sizeof(MESH_INDEX);
		pipeline_layout_create_info.pushConstantRangeCount = 1;
		pipeline_layout_create_info.pPushConstantRanges = &pushConstantRange;
		vkCreatePipelineLayout(device, &pipeline_layout_create_info, 
			nullptr, &pipelineLayout);
	    // Pipeline State... (FINALLY) 
		VkGraphicsPipelineCreateInfo pipeline_create_info = {};
		pipeline_create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeline_create_info.stageCount = 2;
		pipeline_create_info.pStages = stage_create_info;
		pipeline_create_info.pInputAssemblyState = &assembly_create_info;
		pipeline_create_info.pVertexInputState = &input_vertex_info;
		pipeline_create_info.pViewportState = &viewport_create_info;
		pipeline_create_info.pRasterizationState = &rasterization_create_info;
		pipeline_create_info.pMultisampleState = &multisample_create_info;
		pipeline_create_info.pDepthStencilState = &depth_stencil_create_info;
		pipeline_create_info.pColorBlendState = &color_blend_create_info;
		pipeline_create_info.pDynamicState = &dynamic_create_info;
		pipeline_create_info.layout = pipelineLayout;
		pipeline_create_info.renderPass = renderPass;
		pipeline_create_info.subpass = 0;
		pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
		vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, 
			&pipeline_create_info, nullptr, &pipeline);

		/***************** CLEANUP / SHUTDOWN ******************/
		// GVulkanSurface will inform us when to release any allocated resources
		shutdown.Create(vlk, [&]() {
			if (+shutdown.Find(GW::GRAPHICS::GVulkanSurface::Events::RELEASE_RESOURCES, true)) {
				CleanUp(); // unlike D3D we must be careful about destroy timing
			}
		});
	}
	void Render()
	{
		// TODO: Part 2a
		static std::chrono::steady_clock::time_point end;
		std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - now;
		end = now;
		// TODO: Part 4d
		// grab the current Vulkan commandBuffer
		unsigned int currentBuffer;
		vlk.GetSwapchainCurrentImage(currentBuffer);
		VkCommandBuffer commandBuffer;
		vlk.GetCommandBuffer(currentBuffer, (void**)&commandBuffer);
		// what is the current client area dimensions?
		unsigned int width, height;
		win.GetClientWidth(width);
		win.GetClientHeight(height);
		// setup the pipeline's dynamic settings
		VkViewport viewport = {
            0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1
        };
        VkRect2D scissor = { {0, 0}, {width, height} };
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
		

		//MatrixProxy.MultiplyMatrixF()
		//ShaderModelData.ViewMatrix






		// now we can draw
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexHandle, offsets);
		// TODO: Part 1h
		vkCmdBindIndexBuffer(commandBuffer, indiciesBuffer, 0, VK_INDEX_TYPE_UINT32);
		// TODO: Part 4d
		GvkHelper::write_to_buffer(device, VectorDeviceMemory[currentBuffer], &ShaderModelData, sizeof(ShaderModelData));
		// TODO: Part 2i
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet[currentBuffer], 0, nullptr);
		// TODO: Part 3b
		int offSet = 0;
		unsigned outerCount = 1;
		for (std::map<std::string, Mesh_Struct>::iterator it = MeshMap.begin(); it != MeshMap.end(); ++it, offSet++) {
			for (unsigned i = 0; i < it->second.h2bParser.meshCount; i++) {
				MESH_INDEX tempMesh = { MaterialOffset[offSet] + it->second.h2bParser.meshes[i].materialIndex, outerCount };
				vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(MESH_INDEX), &tempMesh);
				vkCmdDrawIndexed(commandBuffer, it->second.h2bParser.meshes[i].drawInfo.indexCount, it->second.WorldMatrices.size(), IndexOffsets[offSet] + it->second.h2bParser.meshes[i].drawInfo.indexOffset, VertexOffsets[offSet], MatrixOffset[offSet]);
			}
		}
	}
	void ChangeScene() {
		vkDeviceWaitIdle(device);
		vkDestroyBuffer(device, indiciesBuffer, nullptr);
		vkFreeMemory(device, indiciesData, nullptr);
		// TODO: Part 2d
		vkDestroyBuffer(device, vertexHandle, nullptr);
		vkFreeMemory(device, vertexData, nullptr);
		Vertexes.clear();
		Indexes.clear();
		MeshMap.clear();
		if (CurrentScene == 0) {
			if (!ParseFile("../GameLevel.txt"))
				exit(69);
			std::cout << "First";
			CurrentScene = 1;
		}
		else {
			if (!ParseFile("../GameLevel.txt"))
				exit(69);
			std::cout << "Second";
			CurrentScene = 0;
		}
		int externalI = 0;
		int externalZ = 0;
		for (std::map<std::string, Mesh_Struct>::iterator it = MeshMap.begin(); it != MeshMap.end(); ++it) {
			for (int y = 0; y < it->second.h2bParser.materialCount; y++, externalI++)
				ShaderModelData.materials[externalI] = it->second.h2bParser.materials[y].attrib;
			for (int y = 0; y < it->second.WorldMatrices.size(); y++, externalZ++)
				ShaderModelData.matricies[externalZ] = it->second.WorldMatrices[y];
		}
		VkPhysicalDevice physicalDevice = nullptr;
		vlk.GetDevice((void**)&device);
		vlk.GetPhysicalDevice((void**)&physicalDevice);
		H2B::VERTEX* tempVec = &Vertexes[0];
		GvkHelper::create_buffer(physicalDevice, device, sizeof(*tempVec) * Vertexes.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &vertexHandle, &vertexData);
		GvkHelper::write_to_buffer(device, vertexData, &*tempVec, sizeof(*tempVec) * Vertexes.size());
		// TODO: Part 1g
		unsigned* tempInd = &Indexes[0];
		GvkHelper::create_buffer(physicalDevice, device, sizeof(*tempInd) * Indexes.size(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &indiciesBuffer, &indiciesData);
		GvkHelper::write_to_buffer(device, indiciesData, &*tempInd, sizeof(*tempInd) * Indexes.size());
	}
	void UpdateCamera() {
		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - now;
		now = end;
		GW::MATH::GMATRIXF viewCopyMatrix = ShaderModelData.ViewMatrix;
		MatrixProxy.InverseF(viewCopyMatrix, viewCopyMatrix);
		// TODO: Part 4c
		// TODO: Part 4d
		const float Camera_Speed = 5.0f;

		float spacebar;
		float SongChange;
		static bool pushIt;
		static bool changeSong;
		static bool changed;
		static bool played;
		Ginput.GetState(23, spacebar);
		Ginput.GetState(62, SongChange);
		if (PLAYMUSIC) {
			if (spacebar != 0)
				pushIt = true;
			else {
				pushIt = false;
				played = false;
			}
			if (SongChange != 0)
				changeSong = true;
			else {
				changeSong = false;
				changed = false;
			}
			bool isplaying = false;
			GSound.isPlaying(isplaying);
			if (!isplaying)
				GMusic.Resume();
			if (pushIt && !played && !isplaying) {
				GMusic.Pause();
				GSound.Create("../PushIt.wav", GAudio);
				GSound.Play();
				pushIt = false;
				played = true;
			}
			if (changeSong && !changed) {
				SongChoice++;
				GMusic.Stop();
				GMusic.Create(Songs[SongChoice % 3], GAudio);
				GMusic.Play();
				changeSong = false;
				changed = true;
			}
		}
		static bool SceneChange;
		static bool scenechanged;
		float R;
		Ginput.GetState(55, R);
		if (R != 0)
			SceneChange = true;
		else {
			SceneChange = false;
			scenechanged = false;
		}
		if (SceneChange && !scenechanged) {
			ChangeScene();
			SceneChange = false;
			scenechanged = true;
		}

		float leftshift;
		Ginput.GetState(14, leftshift);
		bool isConnected = false;
		float y_change = spacebar - leftshift;
		float total_y_change = y_change * Camera_Speed * elapsed_seconds.count();
		GW::MATH::GVECTORF translation = { 0, total_y_change, 0, 1 };
		MatrixProxy.TranslateLocalF(viewCopyMatrix, translation, viewCopyMatrix);
		// TODO: Part 4e
		float PerFrameSpeed = Camera_Speed * elapsed_seconds.count();
		float akey = 0;
		Ginput.GetState(38, akey);
		float ekey = 0;
		Ginput.GetState(42, ekey);
		float qkey = 0;
		Ginput.GetState(54, qkey);
		float wkey = 0;
		Ginput.GetState(60, wkey);
		float skey = 0;
		Ginput.GetState(56, skey);
		float dkey = 0;
		Ginput.GetState(41, dkey);
		float total_z_change = wkey - skey;
		float total_x_change = dkey - akey;
		float total_roll = (qkey - ekey) * elapsed_seconds.count();
		GW::MATH::GVECTORF tvector = { total_x_change * PerFrameSpeed, 0, total_z_change * PerFrameSpeed };
		MatrixProxy.TranslateLocalF(viewCopyMatrix, tvector, viewCopyMatrix);
		//GW::MATH::GMATRIXF tmatrix = { total_x_change * PerFrameSpeed, 0, total_z_change * PerFrameSpeed };
		//Proxy.MultiplyMatrixF(tmatrix, viewCopyMatrix, viewCopyMatrix);
		// TODO: Part 4f
		GW::GReturn returnValue = GW::GReturn::FAILURE;;
		float mouse_y_delta;
		float mouse_x_delta;
		float total_pitch = 0;
		float total_yaw = 0;
		unsigned screenheight;
		unsigned screenwidth;
		float AR = 0;
		vlk.GetAspectRatio(AR);
		win.GetHeight(screenheight);
		win.GetWidth(screenwidth);
		returnValue = Ginput.GetMouseDelta(mouse_x_delta, mouse_y_delta);
		float thumbspeed = 3.1415f * elapsed_seconds.count();
		if (returnValue == GW::GReturn::SUCCESS) {
			total_pitch += 1.13446f * mouse_y_delta / screenheight;
			total_yaw += 1.13446f * AR * mouse_x_delta / screenwidth;
		}
		//Part 4F
		GW::MATH::GMATRIXF PitchMatrix;
		MatrixProxy.IdentityF(PitchMatrix);

		MatrixProxy.RotationYawPitchRollF(0.0f, total_pitch, 0.0f, PitchMatrix);
		MatrixProxy.MultiplyMatrixF(PitchMatrix, viewCopyMatrix, viewCopyMatrix);

		//Part 4G
		GW::MATH::GMATRIXF YawMatrix;
		MatrixProxy.IdentityF(YawMatrix);
		GW::MATH::GVECTORF pos = viewCopyMatrix.row4;
		MatrixProxy.RotationYawPitchRollF(total_yaw, 0.0f, 0.0f, YawMatrix);
		MatrixProxy.MultiplyMatrixF(viewCopyMatrix, YawMatrix, viewCopyMatrix);
		viewCopyMatrix.row4 = pos;
		// TODO: Part 4g
		// TODO: Part 4c
		//ViewMatrix = viewCopyMatrix;
		ShaderModelData.camPos = pos;
		MatrixProxy.InverseF(viewCopyMatrix, ShaderModelData.ViewMatrix);
	}
	
private:
	void CleanUp()
	{
		// wait till everything has completed
		vkDeviceWaitIdle(device);
		// Release allocated buffers, shaders & pipeline
		// TODO: Part 1g
		vkDestroyBuffer(device, indiciesBuffer, nullptr);
		vkFreeMemory(device, indiciesData, nullptr);
		// TODO: Part 2d
		for (int i = 0; i < VectorVkBuffer.size(); i++) {
			vkDestroyBuffer(device, VectorVkBuffer[i], nullptr);
			vkFreeMemory(device, VectorDeviceMemory[i], nullptr);
		}
		VectorVkBuffer.clear();
		VectorDeviceMemory.clear();
		vkDestroyBuffer(device, vertexHandle, nullptr);
		vkFreeMemory(device, vertexData, nullptr);
		vkDestroyShaderModule(device, vertexShader, nullptr);
		vkDestroyShaderModule(device, pixelShader, nullptr);
		// TODO: Part 2e
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
		// TODO: part 2f
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyPipeline(device, pipeline, nullptr);
	}
};
