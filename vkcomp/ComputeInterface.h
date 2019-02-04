/*
Copyright © 2019
Roberto Cavicchioli, Nicola Capodieci

See LICENSE.txt
*/

#ifndef COMPUTE_INTERFACE_H
#define COMPUTE_INTERFACE_H

#include "datatypes.h"
#include "CrossFileAdapter.h"

#define NVIDIA_PREFERRED 0x10DE
#define AMD_PREFERRED 0x1002
#define INTEL_PREFERRED 0x163C
#define ARM_MALI_PREFERRED 0x13B5
#define QUALCOMM_PREFERRED 0x5143

#define HOST_TO_DEVICE 0
#define DEVICE_TO_HOST 1

	class ComputeInterface
	{

	public:
		void createContext();

		virtual void errorCheck()=0;

		std::string fromVendorIDtoString(uint32_t vendorID);

		virtual int32_t loadAndCompileShader(const std::string, const std::string)=0;
		virtual int32_t loadAndCompileShader(CrossFileAdapter, const std::string)=0;
		virtual void printContextInformation();

		virtual void *deviceSideAllocation(const uint64_t size, const BufferUsage buffer_usage, const uint32_t stride=0)=0;

		virtual void setArg(void**, const std::string shader_id, const uint32_t)=0;

		virtual inline void synchLaunch()=0;

		virtual void synchBuffer(void **, const uint8_t) = 0;

		virtual void copySymbolInt(  int value, const  std::string shader, const uint32_t location)=0;

		virtual void copySymbolDouble( double value, const  std::string shader, const uint32_t location)=0;

		virtual void copySymbolFloat( float value, const std::string shader, const uint32_t location)=0;

		virtual void setLaunchConfiguration(const ComputeWorkDistribution_t blocks, const ComputeWorkDistribution_t threads=NULL)=0;

		virtual void launchComputation(const std::string computation_identifier)=0;

		virtual inline void deviceSynch()=0;

		virtual void freeResource(void* resource)=0;

		virtual void freeResources()=0;

		virtual ~ComputeInterface();


	protected:
		LaunchConfiguration_t latest_launch_conf;

	private:
		bool context_created = false;


	};

#endif
