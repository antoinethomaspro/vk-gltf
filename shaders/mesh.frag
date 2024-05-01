#version 450

#extension GL_GOOGLE_include_directive : require
#include "input_structures.glsl"

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 FragPos;
layout (location = 2) in vec3 inColor;
layout (location = 3) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

vec3 lightColor = vec3(1.0, 1.0, 1.0);

vec3 lightPos = vec3(4.0, 10.0, 4.0);

void main() 
{
	float ambientStrength = 0.01;
    vec3 ambient = ambientStrength * lightColor;

	float specularStrength = 0.5;

	vec3 eyePos = sceneData.eyePos;
	vec3 Normal = normalize(inNormal);
	vec3 lightDir = normalize(lightPos - FragPos);  
	vec3 viewDir = normalize(eyePos - FragPos);
	vec3 reflectDir = reflect(-lightDir, Normal);  

	float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
	vec3 specular = specularStrength * spec * lightColor;  

	float diff = max(dot(Normal, lightDir), 0.0);
	vec3 diffuse = diff * lightColor;

	vec3 objectColor = inColor * texture(colorTex,inUV).xyz;
	vec3 result = (ambient + diffuse + specular) * objectColor;

	outFragColor = vec4(result ,1.0f);
}
