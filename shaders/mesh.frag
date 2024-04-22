#version 450

#extension GL_GOOGLE_include_directive : require
#include "input_structures.glsl"

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inColor;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 FragPos;


layout (location = 0) out vec4 outFragColor;

vec3 lightPos = vec3(0.0, 100.0, 0.0);
vec3 objectColor = vec3(1.0);

void main() 
{
	vec3 norm = normalize(inNormal);
	vec3 lightDir = normalize(lightPos - FragPos);  
	float diff = max(dot(norm, lightDir), 0.0);

	vec3 color = inColor * texture(colorTex,inUV).xyz;
	vec3 ambient = color *  sceneData.ambientColor.xyz;

	outFragColor = vec4(diff * objectColor ,1.0f);
}
