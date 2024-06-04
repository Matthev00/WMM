#version 330

in vec3 v_position;
in vec3 v_normal;

out vec4 f_color;

const vec3 light_position = vec3(0.0, 7.0, -15.0);
const vec3 light_ambient = vec3(0.1, 0.1, 0.1);
const vec3 light_diffuse = vec3(1.0, 1.0, 1.0);
const vec3 light_specular = vec3(0.8, 0.8, 0.8);

uniform vec3 material_ambient;
uniform vec3 material_diffuse;
uniform vec3 material_specular;
uniform float material_shininess;

uniform vec3 camera_position;
uniform vec3 color;

void main()
{
	vec3 ambient = light_ambient * material_ambient;

	vec3 N = normalize(v_normal);
	vec3 L = normalize(light_position - v_position);
	float cosNL = clamp(dot(N, L), 0.0, 1.0);

	vec3 diffuse = light_diffuse * material_diffuse * cosNL;

	vec3 V = normalize(camera_position - v_position);
    vec3 R = reflect(-L, N);
    float spec = pow(max(dot(V, R), 0.0), material_shininess);
    vec3 specular = light_specular * material_specular * spec;

    vec3 phong_color = clamp(ambient + diffuse + specular, 0.0, 1.0) * color;

	f_color = vec4(phong_color, 1.0);
}