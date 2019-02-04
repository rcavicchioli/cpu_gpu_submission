/*
Copyright © 2019
Roberto Cavicchioli, Nicola Capodieci

See LICENSE.txt
*/

#ifndef COMMAND_LIST_BASED_H
#define COMMAND_LIST_BASED_H

#include "macrodefs.h"
#include <vector>
#include <map>

#define CMD_LIST_IS_CREATED 0
#define CMD_LIST_IN_CREATION 1
#define CMD_LIST_IS_RESET 2

#define PIPELINE_IS_RESTING 0
#define PIPELINE_IN_CREATION 1

#define PIPELINE_HANDLE uint32_t

class CommandListBased
{
public:
	CommandListBased();

	void startCreatePipeline(std::string shader_id);

	virtual void setSymbol(const uint32_t location, uint32_t byte_width) = 0;

	uint32_t finalizePipeline();

	void startCreateCommandList();
	
	void finalizeCommandList();

	virtual void submitWork()=0;

	void selectPipeline(const uint32_t pipeline_index);

	void updatePointer(void** elem);

	void updatePointers(std::vector<void**> v);

	~CommandListBased();

protected:
	inline bool verifyCmdListState(const uint8_t expectation);

	inline bool verifyPipelineCreationState(const uint8_t expectation);

	std::map<intptr_t, intptr_t> set_to_update;

private:
	uint8_t cmd_list_creation_state;
	uint8_t pipeline_creation_state;
};

#endif
