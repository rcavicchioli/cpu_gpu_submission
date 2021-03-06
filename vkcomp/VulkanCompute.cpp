
/*
Copyright © 2019
Roberto Cavicchioli, Nicola Capodieci

See LICENSE.txt
*/

#include "VulkanCompute.h"
#include "stdafx.h"

//LOOK AT ME WHEN YOU DON'T KNOW HOW OTHER HW WILL DO!!!!
//http://vulkan.gpuinfo.org/
////////////////////////////////////////////////////////

#ifdef DEBUG_VK_ENABLED
PFN_vkCreateDebugReportCallbackEXT fvkCreateDebugReportCallbackEXT = NULL;
PFN_vkDestroyDebugReportCallbackEXT fvkDestroyDebugReportCallbackEXT = NULL;

VKAPI_ATTR VkBool32 VKAPI_CALL
vulkanDebugCallback(VkDebugReportFlagsEXT msg_flags,     //type of error reporting (WARNING, ERROR or something else)
					VkDebugReportObjectTypeEXT obj_type, //this is the type of object that caused the error. Destroying the device before all other stuff is an error. Device is this example case.
					uint64_t src_obj,                    //this is basically the pointer of the said object.
					size_t location,                     //nobody knows for sure
					int32_t msg_code,                    //same as above.
					const char* layer_prefix,            //tells us which layer is related to the error/warning etc...
					const char* msg,                     //the message itself. Pretty much all that we care of
					void* usr_data						 //user data for more refined debug msg processing.
				)
{
	std::cout << msg << std::endl;

	return false; //always return false for some reason I did not understand
}

void VulkanCompute::setupDebug()
{
	fvkCreateDebugReportCallbackEXT =  (PFN_vkCreateDebugReportCallbackEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
	fvkDestroyDebugReportCallbackEXT = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");

	if (fvkCreateDebugReportCallbackEXT != NULL)
		DBG_PRINT("vkCreateDebug ok!")
	else {
		//DBG_PRINT("Your drivers are broken. Trying with GetProcAddress")
		//HMODULE vulkan_module = LoadLibrary(L"vulkan-1.dll");
		//fvkCreateDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT) GetProcAddress(vulkan_module, "vkCreateDebugReportCallbackEXT");
		//if(fvkCreateDebugReportCallbackEXT!=NULL) DBG_PRINT("GetProcAddress did the trick")
		/*else*/ FATAL_EXIT("Unable to fetch PFN for destroying debug callback")
	}

	if (fvkDestroyDebugReportCallbackEXT != NULL)
		DBG_PRINT("vkDestroyDebug ok!")
	else FATAL_EXIT("Unable to fetch PFN for destroying debug callback")


	VkDebugReportCallbackCreateInfoEXT report_cb_info = {};
	report_cb_info.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
	report_cb_info.pNext = NULL;
	report_cb_info.flags = VK_DEBUG_REPORT_INFORMATION_BIT_EXT |
						   VK_DEBUG_REPORT_WARNING_BIT_EXT |
						   VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT |
						   VK_DEBUG_REPORT_ERROR_BIT_EXT |
						   VK_DEBUG_REPORT_DEBUG_BIT_EXT;
	report_cb_info.pfnCallback = vulkanDebugCallback;
	
	fvkCreateDebugReportCallbackEXT(instance, &report_cb_info, NULL, &debug_callback_handle);

}

void VulkanCompute::destroyDebug()
{
	fvkDestroyDebugReportCallbackEXT(instance, debug_callback_handle, NULL);
	debug_callback_handle = NULL;
}

#endif

VulkanCompute::VulkanCompute(std::string path_to_glslc, uint8_t preferred_vendor_, uint64_t device_mem_req) : CommandListBased()
{
	//if (path_to_glslc.length() > 1)
		glslc_folder = path_to_glslc;
	/*else 
	{
		char cCurrentPath[FILENAME_MAX];
		GetCurrentDir(cCurrentPath, sizeof(cCurrentPath));
		glslc_folder = std::string(cCurrentPath) + FILE_SEPARATOR + "vkres" + FILE_SEPARATOR;
	}*/

	last_loaded_shader = "";
	last_loaded_shader_module = NULL;
	preferred_vendor = preferred_vendor_;
	device_memory_requirements = device_mem_req;
	last_operation_error = VK_SUCCESS;
	wip_pipeline_creation = {};
	currently_selected_pipeline_index = -1;
	//buffer_counter = 0;

}

void VulkanCompute::createContext()
{
	ComputeInterface::createContext();

	//pre_allocated_push_constants = (uint8_t*)malloc(PRE_ALLOCATED_BUFFER_FOR_PUSH_CONSTANTS_SIZE);

	VkApplicationInfo app_info = {};
	app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	app_info.apiVersion = VK_MAKE_VERSION(1, 0, 24); //well you tell me... vulkan.gpuinfo
	app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	app_info.pApplicationName = "Vulkan Compute Engine";

	//instance_extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
	//instance_extensions.push_back(PLATFORM_SURFACE_EXTENSION_NAME);

	VkInstanceCreateInfo c_app_info = {};
	c_app_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	c_app_info.ppEnabledExtensionNames = NULL; //instance_extensions.data();
	c_app_info.enabledExtensionCount = 0;      //(uint32_t)instance_extensions.size();
	c_app_info.pApplicationInfo = &app_info;

#ifdef DEBUG_VK_ENABLED

	uint32_t numInstanceExtensions;
	vkEnumerateInstanceExtensionProperties(NULL, &numInstanceExtensions, NULL);
	std::vector<VkExtensionProperties> extensions;
	extensions.resize(numInstanceExtensions);
	vkEnumerateInstanceExtensionProperties(NULL, &numInstanceExtensions, extensions.data());

	dbg_extension_strings.push_back("VK_EXT_debug_report"); //otherwise will fail instance creation

	c_app_info.ppEnabledExtensionNames = dbg_extension_strings.data(); //instance_extensions.data();
	c_app_info.enabledExtensionCount = (uint32_t) dbg_extension_strings.size();      //(uint32_t)instance_extensions.size();

	uint32_t numInstanceLayers;
	vkEnumerateInstanceLayerProperties(&numInstanceLayers, NULL);
	std::vector<VkLayerProperties> layers;
	layers.resize(numInstanceLayers);
	vkEnumerateInstanceLayerProperties(&numInstanceLayers, layers.data());

	/*for (size_t i = 0; i < numInstanceLayers; i++)
		std::cout << layers.at(i).layerName << std::endl;*/

	//things we can use to debug
	//VK_LAYER_RENDERDOC_Capture
	//VK_LAYER_LUNARG_vktrace
	//VK_LAYER_LUNARG_core_validation
	dbg_layer_strings.push_back("VK_LAYER_LUNARG_standard_validation");
	c_app_info.enabledLayerCount = (uint32_t) dbg_layer_strings.size();
	c_app_info.ppEnabledLayerNames = dbg_layer_strings.data();
#else
	c_app_info.enabledLayerCount = 0;
	c_app_info.ppEnabledLayerNames = NULL;
#endif

	phys_device = NULL;

	VK_CRITICAL_CALL(vkCreateInstance(&c_app_info, NULL, &instance);)
	errorCheck();

#ifdef DEBUG_VK_ENABLED
	setupDebug();
#endif

	uint32_t gpus = 0;
	VK_CRITICAL_CALL(vkEnumeratePhysicalDevices(instance, &gpus, NULL);)
	errorCheck();
	std::vector<VkPhysicalDevice> gpu_list(gpus);
	VK_CRITICAL_CALL(vkEnumeratePhysicalDevices(instance, &gpus, gpu_list.data());)
	errorCheck();

	if (gpus == 1 || preferred_vendor == NO_VENDOR_PREFERRED) {
		phys_device = gpu_list.at(0);
		vkGetPhysicalDeviceProperties(phys_device, &phys_device_props);
	}
	else
	{
		for (uint32_t i = 0; i < gpu_list.size(); i++)
		{
			vkGetPhysicalDeviceProperties(gpu_list[i], &phys_device_props);

			if (phys_device_props.vendorID == preferred_vendor)
			{
				phys_device = gpu_list.at(i);
				break;
			}
			DBG_PRINT("Your preferred GPU vendor was not found. You'll get the default one.")
			phys_device = gpu_list.at(0);
		}
	}

	uint32_t family_count;
	vkGetPhysicalDeviceQueueFamilyProperties(phys_device, &family_count, NULL);
	std::vector<VkQueueFamilyProperties> familyProperties(family_count);
	vkGetPhysicalDeviceQueueFamilyProperties(phys_device, &family_count, familyProperties.data());

	uint32_t queue_index_supporting_compute = -1;
	uint32_t queue_index_supporting_transfer = -1;
	uint32_t supports_all_families = -1;

	for (uint32_t i = 0; i < family_count; i++)
	{
		if ((familyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
			(familyProperties[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
			(familyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
			supports_all_families = i;
			break;
		}
		else if ((familyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
			(familyProperties[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
			!(familyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
			queue_index_supporting_compute = i;
			queue_index_supporting_transfer = i;
			break;
		}
		else if ((familyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
			!(familyProperties[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
			!(familyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
			queue_index_supporting_compute = i;
		}
		else if (!(familyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
			(familyProperties[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
			!(familyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
			queue_index_supporting_transfer = i;
		}
		else if ((familyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
			!(familyProperties[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
			(familyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
			queue_index_supporting_compute = i;
		}
		else if (!(familyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
			(familyProperties[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
			(familyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
			queue_index_supporting_transfer = i;
		}

	}

	if (supports_all_families == -1 && queue_index_supporting_transfer!= queue_index_supporting_compute)
		FATAL_EXIT("I am not dealing with this now.")

	if((queue_index_supporting_transfer == -1 || queue_index_supporting_compute == -1) && supports_all_families==-1)
		FATAL_EXIT("Compute Features not supported by your VK device queue(s)")

	queue_family = supports_all_families != 1 ? supports_all_families : queue_index_supporting_compute;

	float priorities[] = { 1.0f }; // one value in this array, as I only want one queue as of now...
	VkDeviceQueueCreateInfo deviceQueueInfo{};
	deviceQueueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	deviceQueueInfo.pQueuePriorities = priorities;
	deviceQueueInfo.queueFamilyIndex = queue_family;
	deviceQueueInfo.queueCount = 1;

	VkDeviceCreateInfo deviceInfo{};
	deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	deviceInfo.queueCreateInfoCount = 1; 
	deviceInfo.pQueueCreateInfos = &deviceQueueInfo;
	deviceInfo.enabledExtensionCount = 0;
	deviceInfo.ppEnabledExtensionNames = NULL;

	VK_CRITICAL_CALL(vkCreateDevice(phys_device, &deviceInfo, NULL, &device);)
	errorCheck();

	/*pc_range = {};
	pc_range.size = PRE_ALLOCATED_BUFFER_FOR_PUSH_CONSTANTS_SIZE;
	pc_range.offset = 0;
	pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;*/

	vkGetDeviceQueue(device, queue_family, 0, &queue);

	vkGetPhysicalDeviceMemoryProperties(phys_device, &mem_props);

	cmd_pool;
	VkCommandPoolCreateInfo pool_info{};
	pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	pool_info.queueFamilyIndex = queue_family;
	pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	VK_CRITICAL_CALL(vkCreateCommandPool(device, &pool_info, NULL, &cmd_pool);)

	VkCommandBufferAllocateInfo cmd_buf_info {};
	cmd_buf_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	cmd_buf_info.commandPool = cmd_pool;
	cmd_buf_info.commandBufferCount = 1; //should be enough
	cmd_buf_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

	VK_CRITICAL_CALL(vkAllocateCommandBuffers(device, &cmd_buf_info, &cmd_bufs);)
	errorCheck();

	//creating the PSB
	VkBufferCreateInfo PSB_info = {};
	PSB_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	PSB_info.size = device_memory_requirements;
	PSB_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

	VK_CRITICAL_CALL(vkCreateBuffer(device, &PSB_info, NULL, &PSB.buffer));
	errorCheck();

	VkMemoryAllocateInfo memAlloc{};
	memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	VkMemoryRequirements memReqs;
	vkGetBufferMemoryRequirements(device, PSB.buffer, &memReqs);

	memAlloc.allocationSize = memReqs.size;

	memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, &mem_props);
	
	if (memAlloc.memoryTypeIndex == -1)
		FATAL_EXIT("Error in getting device memory type")

	VK_CRITICAL_CALL(vkAllocateMemory(device, &memAlloc, NULL, &PSB.memory);)
	errorCheck();

	VK_CRITICAL_CALL(vkBindBufferMemory(device, PSB.buffer, PSB.memory, 0);)
	errorCheck();

	VK_CRITICAL_CALL(vkMapMemory(device, PSB.memory, 0, memAlloc.allocationSize, 0, (void**)&PSB_data);)
	errorCheck();

	PSB.buffer_info.buffer = PSB.buffer;
	PSB.buffer_info.offset = 0;
	PSB.buffer_info.range = device_memory_requirements;

	submit_info = {};
	submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers = &cmd_bufs;

	DBG_PRINT("Congratulations! You have survived VK init without crashes!")

}

void VulkanCompute::printContextInformation()
{
	std::cout << "Device ID : " << phys_device_props.deviceID << std::endl;
	std::cout << "Device Name : " << phys_device_props.deviceName << std::endl;
	std::cout << "Vendor id : " << fromVendorIDtoString(phys_device_props.vendorID).c_str() << std::endl;
	std::cout << "Device type : " << fromVKDeviceTypeToString(phys_device_props.deviceType) << std::endl;
	std::cout << "API Version : " << ((phys_device_props.apiVersion >> 22) & 0x3FF) << "." <<
									 ((phys_device_props.apiVersion >> 12) & 0x3FF) << "." << 
									 ((phys_device_props.apiVersion & 0xFFF)) << std::endl;
	std::cout << "Folder to look for glslValidator " << glslc_folder.c_str() << std::endl;
}

int32_t VulkanCompute::loadAndCompileShader(CrossFileAdapter f, const std::string shader_id)
{
	std::string compilation_output = "";
	std::string cmd_line = "glslangValidator";

#ifdef _WIN32
	cmd_line += ".exe";
#endif

	size_t index = f.getAbsolutePath().find_last_of(FILE_SEPARATOR);
	if (index != std::string::npos) {
#ifdef _WIN32
		index++;
#endif
	}
	else index = 0;

	std::string out_name = f.getAbsolutePath().substr(index, f.getAbsolutePath().length()) + ".spv";

	std::cout << cmd_line + " -V " + f.getAbsolutePath() + " -o " + f.getAbsolutePath().substr(0, index) + out_name << std::endl;
	compilation_output = exec(cmd_line + " -V " + f.getAbsolutePath() + " -o " + f.getAbsolutePath().substr(0, index) + out_name);
	//compilation_output = exec(cmd_line + " -V " + f.getAbsolutePath() + " -o " + glslc_folder + FILE_SEPARATOR + out_name);

	//std::cout << compilation_output.c_str() << std::endl;

	uint32_t errors = 0;

	if (compilation_output.find("Error") != std::string::npos || compilation_output.find("error") != std::string::npos)
		errors++;

	std::string out_filename = f.getAbsolutePath().substr(0, index) + out_name;

	//std::cout << "SPIR-V generated in : " << out_filename.c_str() << std::endl;

	if (errors > 0) {
		std::cout << "VK compile error : " << std::endl << compilation_output << std::endl;
		return -1;
	}

	//std::cout << "Compilation output : " << std::endl << compilation_output << std::endl;

	size_t size;
	FILE *fp = fopen(out_filename.c_str(), "rb");

	if (fp == nullptr) {
		std::cout << "Error in opening file " << out_filename.c_str() << std::endl;
		return -1;
	}

	fseek(fp, 0L, SEEK_END);
	size = ftell(fp);
	fseek(fp, 0L, SEEK_SET);

	char *shaderCode = new char[size];
	size_t retval = fread(shaderCode, size, 1, fp);

	if (retval != 1 || size<0)
		FATAL_EXIT("Unable to parse specified SPIR-V related file")

		fclose(fp);

	VkShaderModule shaderModule;
	VkShaderModuleCreateInfo moduleCreateInfo;
	moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	moduleCreateInfo.pNext = NULL;
	moduleCreateInfo.codeSize = size;
	moduleCreateInfo.pCode = (uint32_t*)shaderCode;
	moduleCreateInfo.flags = 0;

	VK_CRITICAL_CALL(vkCreateShaderModule(device, &moduleCreateInfo, VK_NULL_HANDLE, &shaderModule);)
		errorCheck();

	delete[] shaderCode;

	program_map.insert(std::pair<std::string, VkShaderModule>(shader_id, shaderModule));

	DBG_PRINT("Shader compiled successfully")

	return (int32_t)program_map.size();

}

int32_t VulkanCompute::loadAndCompileShader(const std::string s, const std::string shader_id)
{

	std::string filename = glslc_folder + shader_id + "_vk.comp";
	std::ofstream out(filename);
	out << s.c_str();
	out.close();

	return loadAndCompileShader(CrossFileAdapter(filename.c_str()),shader_id);
}

uint32_t VulkanCompute::getMemoryType(uint32_t typeBits, VkFlags properties, const VkPhysicalDeviceMemoryProperties *const mem_props)
{
	if (mem_props == NULL)
	{
		FATAL_EXIT("Null physical device mem props instance...")
	}

	for (uint32_t i = 0; i < 32; i++)
	{
		if ((typeBits & 1) == 1)
		{
			if ((mem_props->memoryTypes[i].propertyFlags & properties) == properties)
			{
				return i;
			}
		}
		typeBits >>= 1;
	}
	
	return -1;
}

void *VulkanCompute::deviceSideAllocation(const uint64_t size, const BufferUsage buffer_usage, const uint32_t stride)
{

	uint32_t usage;

	if (buffer_usage == BufferUsage::BUF_IN)
		usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	else if (buffer_usage == BufferUsage::BUF_OUT)
		usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	else usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

	VkBufferCreateInfo buffer_info = {};
	buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.size = size;
	buffer_info.usage = usage;

	VulkanBufferStruct out_buf;

	VK_CRITICAL_CALL(vkCreateBuffer(device, &buffer_info, NULL, &out_buf.buffer);)
	errorCheck();
	VkMemoryRequirements memReqs;
	vkGetBufferMemoryRequirements(device, out_buf.buffer, &memReqs);
	VkMemoryAllocateInfo memAlloc = {};
	memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memAlloc.allocationSize = memReqs.size;
	memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &mem_props);
	VK_CRITICAL_CALL(vkAllocateMemory(device, &memAlloc, NULL, &out_buf.memory);)
	errorCheck();
	VK_CRITICAL_CALL(vkBindBufferMemory(device, out_buf.buffer, out_buf.memory, 0);)
	errorCheck();

	intptr_t newadd;

	if (PSB_allocation_table.size() == 0) 
		newadd = (intptr_t)PSB_data;
	else newadd = PSB_allocation_table.rbegin()->first + PSB_allocation_table.rbegin()->second;

	out_buf.buffer_info.buffer = out_buf.buffer;
	out_buf.buffer_info.offset = 0;
	out_buf.buffer_info.range = size;

	PSB_allocation_table.insert(std::pair<intptr_t, uint64_t>(newadd, size));

	device_side_allocation_table.insert(std::pair<intptr_t,VulkanBufferStruct>(newadd,out_buf));

	return (void*)newadd;
}

void VulkanCompute::startCreatePipeline(std::string shader_id)
{
	CommandListBased::startCreatePipeline(shader_id);

	wip_spec_info = {};
	wip_spec_info.dataSize = 0; //to be filled just before creating the pipeline.
	wip_spec_info.mapEntryCount = 3; //to be updated just before creating the pipeline
	wip_spec_info.pData = pre_allocated_spec_constants;

	if (last_loaded_shader != shader_id) {
		program_iter it = program_map.find(shader_id);
		if (it == program_map.end())
			FATAL_EXIT("Non existent shader selected")

		last_loaded_shader = shader_id;
		last_loaded_shader_module = it->second;
	}

	VkComputePipelineCreateInfo pipeline_create_info = {};
	
	pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pipeline_create_info.pNext = NULL;
	pipeline_create_info.flags = VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
	pipeline_create_info.basePipelineIndex = 0;
	pipeline_create_info.basePipelineHandle = NULL;
	

	VkPipelineShaderStageCreateInfo shader_stage_info = {};

	shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shader_stage_info.pNext = NULL;
	shader_stage_info.flags = 0;
	shader_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shader_stage_info.module = last_loaded_shader_module;
	shader_stage_info.pName = "main";
	shader_stage_info.pSpecializationInfo = &wip_spec_info;

	pipeline_create_info.stage = shader_stage_info;

	VkPipelineLayoutCreateInfo layout = {};

	layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	layout.pNext = NULL;
	layout.flags = 0;
	layout.setLayoutCount = 1;
	layout.pushConstantRangeCount = 0; //was 1

	wip_pipeline_creation.wip_pipeline = pipeline_create_info;
	wip_pipeline_creation.wip_layout = layout;
	wip_pipeline_creation.size_of_push_constants = 0; //we initially assume no push contants.
	wip_pipeline_creation.shader_id = shader_id;
	wip_pipeline_creation.write_descriptor_sets.clear();
	wip_pipeline_creation.set_layout_bindings.clear();
}

void VulkanCompute::setArg(void **data, const std::string shader_id, const uint32_t index)
{
	if (!CommandListBased::verifyPipelineCreationState(PIPELINE_IN_CREATION))
		FATAL_EXIT("Operation is illegal outside of a startCreatePipeline - finalizePipeline block")

	intptr_t address = (intptr_t)(*data);
	auto iter = device_side_allocation_table.find(address); 

	if (iter == device_side_allocation_table.end())
		FATAL_EXIT("Non-existent buffer")


	VkDescriptorSetLayoutBinding layout_binding = {};
	layout_binding.binding = index;
	layout_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	layout_binding.descriptorCount = 1;
	layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	layout_binding.pImmutableSamplers = NULL;

	wip_pipeline_creation.set_layout_bindings.push_back(layout_binding);

	VkWriteDescriptorSet wr_descr_set = {};

	wr_descr_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	wr_descr_set.descriptorCount = 1;
	wr_descr_set.dstBinding = index;
	wr_descr_set.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	wr_descr_set.pBufferInfo = &(iter->second.buffer_info);															

	wip_pipeline_creation.write_descriptor_sets.push_back(wr_descr_set);

}

void VulkanCompute::setSymbol(const uint32_t location, uint32_t byte_width)
{
	if (!verifyPipelineCreationState(PIPELINE_IN_CREATION))
		FATAL_EXIT("Setting symbols is illegal outside a pipeline creation block")

	auto iter = pc_offset_map.find(wip_pipeline_creation.shader_id);

	if (iter == pc_offset_map.end())
	{
		std::unordered_map<uint32_t,uint32_t> m;
		m.insert(std::pair<uint32_t, uint32_t>(location, 0));
		pc_offset_map.insert(std::pair<std::string,std::unordered_map<uint32_t,uint32_t>>(wip_pipeline_creation.shader_id,m));
	}
	else
	{
		iter->second.insert(std::pair<uint32_t, uint32_t>(location, wip_pipeline_creation.size_of_push_constants));
	}

	wip_pipeline_creation.size_of_push_constants += byte_width;
	
}

uint32_t VulkanCompute::finalizePipeline()
{
	uint32_t outindex = CommandListBased::finalizePipeline();

	wip_spec_info.dataSize = wip_specs_map.at(wip_specs_map.size()-1).offset + wip_specs_map.at(wip_specs_map.size() - 1).size;
	wip_spec_info.mapEntryCount = (uint32_t)wip_specs_map.size();
	wip_spec_info.pMapEntries = wip_specs_map.data(); //do this here and nowhere else

	VkDescriptorPoolSize pool_size = {};
	pool_size.descriptorCount = (uint32_t) wip_pipeline_creation.set_layout_bindings.size();
	pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

	VkDescriptorPoolCreateInfo pool_info = {};
	pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.maxSets = 1; //check this. 
	pool_info.flags = 0;
	pool_info.pNext = NULL;
	pool_info.poolSizeCount = 1;
	pool_info.pPoolSizes = &pool_size;

	VkDescriptorPool descr_pool;

	VK_CRITICAL_CALL(vkCreateDescriptorPool(device, &pool_info, NULL, &descr_pool);)
	errorCheck();

	VkPushConstantRange local_pc_range = {};
	if (wip_pipeline_creation.size_of_push_constants > 0) {
		local_pc_range.size = wip_pipeline_creation.size_of_push_constants;
		local_pc_range.offset = 0;
		local_pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		wip_pipeline_creation.wip_layout.pPushConstantRanges = &local_pc_range;
		wip_pipeline_creation.wip_layout.pushConstantRangeCount = 1;
	}

	VkDescriptorSetLayoutCreateInfo layout_create_info = {};
	layout_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layout_create_info.bindingCount = (uint32_t) wip_pipeline_creation.set_layout_bindings.size();
	layout_create_info.pBindings = wip_pipeline_creation.set_layout_bindings.data();
	layout_create_info.flags = 0;
	layout_create_info.pNext = NULL;
	
	VkDescriptorSetLayout descr_set_layout;

	VK_CRITICAL_CALL(vkCreateDescriptorSetLayout(device, &layout_create_info, NULL, &descr_set_layout);)
	errorCheck();

	wip_pipeline_creation.wip_layout.pSetLayouts = &descr_set_layout;

	VK_CRITICAL_CALL(vkCreatePipelineLayout(device, &(wip_pipeline_creation.wip_layout), NULL, &(wip_pipeline_creation.wip_pipeline.layout));)
	errorCheck();

	VkDescriptorSetAllocateInfo alloc_info = {};

	alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	alloc_info.descriptorPool = descr_pool;
	alloc_info.descriptorSetCount = 1;
	alloc_info.pSetLayouts = &descr_set_layout;

	VkDescriptorSet descr_set;

	VK_CRITICAL_CALL(vkAllocateDescriptorSets(device, &alloc_info, &descr_set);)
	errorCheck();

	for (size_t i = 0; i < wip_pipeline_creation.set_layout_bindings.size(); i++)
		wip_pipeline_creation.write_descriptor_sets.at(i).dstSet = descr_set;

	vkUpdateDescriptorSets(device, (uint32_t) wip_pipeline_creation.write_descriptor_sets.size(), wip_pipeline_creation.write_descriptor_sets.data(), 0, NULL);

	VkPipeline pipeline;

	VK_CRITICAL_CALL(vkCreateComputePipelines(device, NULL, 1, &(wip_pipeline_creation.wip_pipeline), NULL, &pipeline);)
	errorCheck();

	ProducedVKPipeline produced_pipeline;

	produced_pipeline.pipeline = pipeline;
	produced_pipeline.layout = wip_pipeline_creation.wip_pipeline.layout;
	produced_pipeline.descriptor_set = descr_set;
	produced_pipeline.descr_pool = descr_pool;
	produced_pipeline.launch_configuration = latest_launch_conf;
	produced_pipeline.shader_id = wip_pipeline_creation.shader_id;

	outindex = (uint32_t)pipeline_table.size();

	pipeline_table.insert(std::pair<uint32_t,ProducedVKPipeline>(outindex, produced_pipeline));
	
	return outindex;
	
}

void VulkanCompute::startCreateCommandList()
{
	CommandListBased::startCreateCommandList();

	VkCommandBufferBeginInfo info = {};
	info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	info.pNext = NULL;

	VK_CRITICAL_CALL(vkBeginCommandBuffer(cmd_bufs, &info);)
	errorCheck();

}

void VulkanCompute::selectPipeline(const uint32_t pipe_index)
{
	CommandListBased::selectPipeline(pipe_index);

	pipeline_iter pipiter = pipeline_table.find(pipe_index);

	if (pipiter == pipeline_table.end())
		FATAL_EXIT("Invalid pipeline index")

	currently_selected_pipeline_index = pipe_index;

	vkCmdBindPipeline(cmd_bufs, VK_PIPELINE_BIND_POINT_COMPUTE, pipiter->second.pipeline);
	vkCmdBindDescriptorSets(cmd_bufs, VK_PIPELINE_BIND_POINT_COMPUTE, pipiter->second.layout, 0, 1, &(pipiter->second.descriptor_set), 0, NULL);

}

void VulkanCompute::finalizeCommandList()
{
	CommandListBased::finalizeCommandList();

	VK_CRITICAL_CALL(vkEndCommandBuffer(cmd_bufs));
	errorCheck();

}

void VulkanCompute::submitWork()
{

	VK_CRITICAL_CALL(vkQueueSubmit(queue, 1, &submit_info, NULL);) 
	
}

void VulkanCompute::synchBuffer(void ** data, const uint8_t direction)
{
	if(!verifyCmdListState(CMD_LIST_IN_CREATION))
		FATAL_EXIT("Buffer transfer operation not allowed outside a command list creation block")

	intptr_t address = (intptr_t)(*data);
	auto iter = device_side_allocation_table.find(address); 

	if (iter == device_side_allocation_table.end())
		FATAL_EXIT("Non-existent buffer")

	uint64_t start_offset = 0;
	uint64_t size = 0;
	size_t i = 0;

	for (const auto &iter_ : PSB_allocation_table) {


		size = iter_.second;

		if (iter_.first == address) {
			break;
		}

		start_offset += iter_.second;

		i++;
	}

	VkBufferCopy copy_region;
	copy_region.size = size;


	if (direction == HOST_TO_DEVICE) {
		copy_region.srcOffset = start_offset; 
		copy_region.dstOffset = 0;
	}
	else {
		copy_region.srcOffset = 0;
		copy_region.dstOffset = start_offset;
	}

	VkMemoryBarrier memoryBarrier;
	memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	memoryBarrier.pNext = NULL;
	VkPipelineStageFlags srcFlag;
	VkPipelineStageFlags dstFlag;
	
	if (direction == HOST_TO_DEVICE) {
		memoryBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
		memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		srcFlag = VK_PIPELINE_STAGE_TRANSFER_BIT;
		dstFlag = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
		vkCmdCopyBuffer(cmd_bufs, PSB.buffer, iter->second.buffer, 1, &copy_region);
	}
	else {
		memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		srcFlag = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
		dstFlag = VK_PIPELINE_STAGE_TRANSFER_BIT;
		vkCmdCopyBuffer(cmd_bufs, iter->second.buffer, PSB.buffer, 1, &copy_region);
	}

	vkCmdPipelineBarrier(
				cmd_bufs,
				srcFlag,
				dstFlag,
				0, //no flags
				1, &memoryBarrier,
				0, NULL,
				0, NULL);

}

inline void VulkanCompute::synchLaunch()
{
	if (!verifyCmdListState(CMD_LIST_IN_CREATION))
		FATAL_EXIT("Operation not allowed outside a command list creation block")

	DBG_PRINT("You cannot Synch like that. If you want to synch with the host, consider using fences or events")

}

template<typename T>
inline void VulkanCompute::setUpPushConstant(T value, const uint32_t location)
{

	if(!verifyCmdListState(CMD_LIST_IN_CREATION))
		FATAL_EXIT("Symbols cannot be pushed outside a command list creation block")

	pipeline_iter pipiter = pipeline_table.find(currently_selected_pipeline_index);

	if (pipiter == pipeline_table.end())
		FATAL_EXIT("Invalid pipeline index")

	auto pc_iter = pc_offset_map.find(pipiter->second.shader_id);

	if (pc_iter == pc_offset_map.end())
		FATAL_EXIT("Invalid shader name id during pushing constant operation")

	auto off_iter = pc_iter->second.find(location);

	if (off_iter == pc_iter->second.end())
		FATAL_EXIT("No push contants have been set for the specified location")

	vkCmdPushConstants(cmd_bufs, pipiter->second.layout, VK_SHADER_STAGE_COMPUTE_BIT, off_iter->second, sizeof(T), &value);
			
}

void VulkanCompute::copySymbolInt(int value, const std::string shader, const uint32_t location)
{
		setUpPushConstant(value, location);
}

void VulkanCompute::copySymbolDouble(double value, const std::string shader, const uint32_t location)
{
		setUpPushConstant(value, location);
}

void VulkanCompute::copySymbolFloat(float value, const std::string shader, const uint32_t location)
{
		setUpPushConstant(value, location);
}

void VulkanCompute::setLaunchConfiguration(const ComputeWorkDistribution_t blocks, const ComputeWorkDistribution_t threads)
{

	if(!verifyPipelineCreationState(PIPELINE_IN_CREATION))
		FATAL_EXIT("You cannot set the launch configuration outside of a pipeline creation block")

	wip_specs_map.clear();

	latest_launch_conf.groups_or_blocks = blocks; //we will use this when we dispatch

	pre_allocated_spec_constants[0] = threads.x;
	pre_allocated_spec_constants[1] = threads.y;
	pre_allocated_spec_constants[2] = threads.z;

	VkSpecializationMapEntry entryX = {};
	entryX.constantID = 1;
	entryX.offset = 0;
	entryX.size = sizeof(uint32_t);

	VkSpecializationMapEntry entryY = {};
	entryY.constantID = 2;
	entryY.offset = 1* sizeof(uint32_t);
	entryY.size = sizeof(uint32_t);

	VkSpecializationMapEntry entryZ = {};
	entryZ.constantID = 3;
	entryZ.offset = 2* sizeof(uint32_t);
	entryZ.size = sizeof(uint32_t);

	wip_specs_map.push_back(entryX);
	wip_specs_map.push_back(entryY);
	wip_specs_map.push_back(entryZ);

}

void VulkanCompute::launchComputation(const std::string computation_identifier)
{
	if (!verifyCmdListState(CMD_LIST_IN_CREATION))
		FATAL_EXIT("Launch operation not allowed outside a command list creation block")

	pipeline_iter pipiter = pipeline_table.find(currently_selected_pipeline_index);

	if (pipiter == pipeline_table.end())
		FATAL_EXIT("Invalid pipeline index")

	LaunchConfiguration_t launch_conf = pipiter->second.launch_configuration;

	vkCmdDispatch(cmd_bufs, launch_conf.groups_or_blocks.x, launch_conf.groups_or_blocks.y, launch_conf.groups_or_blocks.z);

	VkMemoryBarrier memoryBarrier;
	memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	memoryBarrier.pNext = NULL;
	memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

	vkCmdPipelineBarrier(
				cmd_bufs,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				0, //no flags
				1, &memoryBarrier,
				0, NULL,
				0, NULL);


}

inline void VulkanCompute::deviceSynch()
{
	if (verifyCmdListState(CMD_LIST_IN_CREATION))
	{
		DBG_PRINT("Synch instruction ignored. Please move this instruction out of a pipeline/cmd list creation block")
		return;
	}

	VK_WFI
}

void VulkanCompute::freeResource(void * resource)
{

	if (set_to_update.size()>0)
		FATAL_EXIT("Previously, you did not release a head pointer, hence, you must call updatePointers before further releasing resources")

	buffer_iterator iter_gpu = device_side_allocation_table.find((intptr_t)resource);
	iter_PSB iter_PSB_ = PSB_allocation_table.find((intptr_t)resource);

	if (iter_gpu == device_side_allocation_table.end() || iter_PSB_ == PSB_allocation_table.end())
		FATAL_EXIT("It is illegal to free a non-existent buffer")

	if (iter_gpu->second.buffer != NULL) {//destroy GPU allocation
		vkDestroyBuffer(device, iter_gpu->second.buffer, NULL);
		vkFreeMemory(device, iter_gpu->second.memory, NULL);
	}

	//special case. The buffer to erase is the last one
	if (iter_PSB_->first == PSB_allocation_table.rbegin()->first)
	{
		DBG_PRINT("Erasing last element of the PSB..")

		PSB_allocation_table.erase(iter_gpu->first);
		device_side_allocation_table.erase(iter_PSB_->first);

		return;
	}

	bool found = false;
	uint64_t size_to_decrement = iter_PSB_->second;

	std::vector<std::pair<intptr_t, VulkanBufferStruct>> temp_dsa;
	std::vector<std::pair<intptr_t, uint64_t>> temp_psb;

	std::pair<intptr_t, uint64_t> start_index_to_memcpy;
	std::pair<intptr_t, uint64_t> end_index_to_memcpy;

	for (iter_gpu = device_side_allocation_table.begin(); iter_gpu != device_side_allocation_table.end(); iter_gpu++)
	{
		if (((intptr_t)resource) == iter_gpu->first) {
			found = true;
			start_index_to_memcpy.first = iter_gpu->first;
			start_index_to_memcpy.second = PSB_allocation_table.find(start_index_to_memcpy.first)->second;
			set_to_update.insert(std::pair<intptr_t, intptr_t>(iter_gpu->first, 0));
			continue;
		}
	
		if (found) {

			intptr_t add_to_modify = iter_gpu->first;
			std::pair<intptr_t,VulkanBufferStruct> newpair_dsa( add_to_modify - size_to_decrement, iter_gpu->second );
			iter_PSB_ = PSB_allocation_table.find(iter_gpu->first);
			std::pair<intptr_t, uint64_t> newpair_psb(add_to_modify - size_to_decrement, iter_PSB_->second );
			temp_dsa.push_back(newpair_dsa);
			temp_psb.push_back(newpair_psb);

			set_to_update.insert(std::pair<intptr_t, intptr_t>(add_to_modify, (add_to_modify - size_to_decrement)));

			end_index_to_memcpy.first = add_to_modify;
			end_index_to_memcpy.second = iter_PSB_->second;

		}
		else
		{
			iter_PSB iter_PSB__ = PSB_allocation_table.find(iter_gpu->first);
			std::pair<intptr_t, VulkanBufferStruct> newpair_dsa(iter_gpu->first, iter_gpu->second);
			std::pair<intptr_t, uint64_t> newpair_psb(iter_PSB__->first, iter_PSB__->second);
			temp_dsa.push_back(newpair_dsa);
			temp_psb.push_back(newpair_psb);
		}

	}

	memmove(((uint8_t*)start_index_to_memcpy.first),
			((uint8_t*)(start_index_to_memcpy.first + start_index_to_memcpy.second)),
			(end_index_to_memcpy.first + end_index_to_memcpy.second) - (start_index_to_memcpy.first + start_index_to_memcpy.second));

	device_side_allocation_table.clear();
	PSB_allocation_table.clear();

	for (size_t i = 0; i < temp_dsa.size(); i++)
	{
		device_side_allocation_table.insert(temp_dsa.at(i));
		PSB_allocation_table.insert(temp_psb.at(i));
	}


}

void VulkanCompute::freeResources()
{
	VK_WFI

	for (iter_programs it = program_map.begin(); it != program_map.end(); it++)
		if (it->second != NULL) 
			vkDestroyShaderModule(device, it->second, NULL);

	program_map.clear();
	last_loaded_shader = "";
	last_loaded_shader_module = NULL;

	for (pipeline_iter it = pipeline_table.begin(); it != pipeline_table.end(); it++) {
		if (it->second.descr_pool != NULL)
			vkDestroyDescriptorPool(device, it->second.descr_pool, NULL);
		if (it->second.layout != NULL)
			vkDestroyPipelineLayout(device, it->second.layout, NULL);
		if (it->second.pipeline != NULL)
			vkDestroyPipeline(device, it->second.pipeline, NULL);
	}

	pipeline_table.clear();
	wip_pipeline_creation = {};
	currently_selected_pipeline_index = -1;

	memset(pre_allocated_spec_constants, 0, 3 * sizeof(uint32_t));
	wip_spec_info = {};
	wip_specs_map.clear();

	pc_offset_map.clear();

	for (buffer_iterator it = device_side_allocation_table.begin(); it != device_side_allocation_table.end(); it++) 
		if (it->second.buffer != NULL) {
			vkDestroyBuffer(device, it->second.buffer, NULL);
			vkFreeMemory(device, it->second.memory, NULL);
		}

	device_side_allocation_table.clear();
	PSB_allocation_table.clear();
	
	vkResetCommandBuffer(cmd_bufs, 0);

	set_to_update.clear();

}

const char * VulkanCompute::fromVKDeviceTypeToString(const VkPhysicalDeviceType dev_type)
{
	switch (dev_type) {
	case		VK_PHYSICAL_DEVICE_TYPE_OTHER: return "Other not recognized device";
	case		VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "Integrated GPU";
	case		VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "Discrete GPU";
	case		VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "Virtual discrete GPU";
	case		VK_PHYSICAL_DEVICE_TYPE_CPU: return "CPU";
	default: return "Unknown";
	}
}

void VulkanCompute::errorCheck()
{
	if (last_operation_error == VK_SUCCESS) return;

	switch (last_operation_error)
	{
		case	VK_ERROR_OUT_OF_HOST_MEMORY: FATAL_EXIT("Host out of memory")
		case	VK_ERROR_OUT_OF_DEVICE_MEMORY: FATAL_EXIT("Device out of memory")
		case	VK_ERROR_INITIALIZATION_FAILED: FATAL_EXIT("Init failed")
		case	VK_ERROR_DEVICE_LOST: FATAL_EXIT("Device is lost")
		case	VK_ERROR_MEMORY_MAP_FAILED: FATAL_EXIT("Memory map failed")
		case	VK_ERROR_LAYER_NOT_PRESENT: FATAL_EXIT("Layer not present")
		case	VK_ERROR_EXTENSION_NOT_PRESENT: FATAL_EXIT("Extension not present")
		case	VK_ERROR_FEATURE_NOT_PRESENT: FATAL_EXIT("Feature not present")
		case	VK_ERROR_INCOMPATIBLE_DRIVER: FATAL_EXIT("Driver not compatible")
		case	VK_ERROR_TOO_MANY_OBJECTS: FATAL_EXIT("Too many objects")
		case	VK_ERROR_FORMAT_NOT_SUPPORTED: FATAL_EXIT("Format not supported")
		case	VK_ERROR_FRAGMENTED_POOL: FATAL_EXIT("Fragmented pool error")
		case    VK_ERROR_INVALID_SHADER_NV: FATAL_EXIT("NV specific error: INVALID SHADER.")
		default : FATAL_EXIT("Unknown error. Define DEBUG_VK_ENABLED to find out more.")
	}

}

VulkanCompute::~VulkanCompute()
{
	VK_WFI

#ifdef DEBUG_VK_ENABLED
		destroyDebug();
#endif

	freeResources();
	vkDestroyCommandPool(device, cmd_pool, NULL);
	vkUnmapMemory(device, PSB.memory);
	vkDestroyBuffer(device, PSB.buffer, NULL);
	vkFreeMemory(device, PSB.memory, NULL);
	vkDestroyDevice(device, NULL);
	vkDestroyInstance(instance, NULL);

}
