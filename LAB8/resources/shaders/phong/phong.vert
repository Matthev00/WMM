#version 330

in vec3 in_position;
in vec3 in_normal;

out vec3 v_position;
out vec3 v_normal;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main()
{

	v_position = (model * vec4(in_position, 1.0)).xyz;
	v_normal = (model * vec4(in_normal, 0.0)).xyz;

	// TODO: Write transformation part
    gl_Position = projection * view * model * vec4(in_position, 1.0);
}