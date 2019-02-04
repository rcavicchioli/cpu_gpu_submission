/*
Copyright © 2019
Roberto Cavicchioli, Nicola Capodieci

See LICENSE.txt
*/

#ifndef CROSS_FILE_ADAPTER_H
#define CROSS_FILE_ADAPTER_H

#include <fstream>
#include "stdafx.h"

class CrossFileAdapter
{
public:
	CrossFileAdapter(const char *arg="");
	std::ifstream getIfStream();

#ifdef _WIN32
	LPCWSTR getLPCWSTRpath();
#endif

	std::string getAbsolutePath();

	void setAbsolutePath(std::string path);

	~CrossFileAdapter();

private:

#ifdef _WIN32
	LPCWSTR windows_style_file_path;
#endif

	std::string abs_path;

};
#endif
