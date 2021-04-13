//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
//=============================================================================================
// Computer Graphics Sample Program: GPU ray casting
//=============================================================================================
//=============================================================================================
// Computer Graphics Sample Program: Ray-tracing-let
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

const float epsilon = 0.0001f;

enum MaterialType { ROUGH, REFLECTIVE };

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	vec3 f0;
	MaterialType type;
	Material(MaterialType t) { type = t; }
};

struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
	};
};

vec3 operator/(vec3 num, vec3 denom) {
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

struct ReflectiveMaterial : Material {
	ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE) {
		vec3 one(1, 1, 1);
		f0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
	}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
	virtual bool isRoom() = 0;
};

class Circle {
	vec3 center;
	float radius;
	std::vector<vec3> points;
public:
	Circle(vec3 center = vec3(0, 0, 0), float r = 0.0f) : center(center), radius(r) { setpoints(); }
	float getRadius() { return radius; }
	float getAreaOfPoint() { return (radius * radius * M_PI) / points.size(); }

	void setpoints() {
		for (int i = 1; i <= 50; i++) {
			float x, y;
			x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX / radius);
			y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX / radius);
			points.push_back(vec3(x, y, center.z));
		}
		for (int i = 0; i < points.size(); i++) {
			if (length(points[i] - center) > radius + epsilon)
				points.erase(points.begin() + i);
		}
	}

	std::vector<vec3> getpoints() { return points; }
};

class Plane {
	vec3 point;
	vec3 normal;
public:
	Plane(vec3 point = vec3(0, 0, 0), vec3 normal = vec3(0, 0, 0)) : point(point), normal(normal) {}
	float f(const vec3 p) {
		return dot(normal, p - point);
	}
};

class Quadric : public Intersectable {
protected:
	mat4 matrix;
	std::vector<Plane> p;
	bool cut, transf;
	mat4 transform, transformInverse, transpose;
public:
	vec3 center;
	Quadric(vec3 center, std::vector<Plane> p, mat4 mat) : center(center), p(p), matrix(mat) { cut = true; transf = false; }
	Quadric(vec3 center, std::vector<Plane> p) : center(center), p(p) { cut = true; transf = false; }
	Quadric(vec3 center) : center(center) { cut = false; transf = false; }

	float f(vec4 r) {

		return dot(r * matrix, r);
	}

	void translate(vec3 transformv) {
		transf = true;
		transform = TranslateMatrix(transformv);
		transformv = transformv * (-1);
		transformInverse = TranslateMatrix(transformv);		//inverse matrix
		for (int i = 0; i < 3; i++) {
			if (transformInverse[3][i] > (epsilon * (-1)) && transformInverse[3][i] < epsilon)
				transformInverse[3][i] = transformInverse[3][i] * (-1);
		}
		float mattoTranspose[4][4];
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				mattoTranspose[i][j] = transformInverse[i][j];
			}
		}
		std::vector<vec4> columns;
		for (int i = 0; i < 4; i++) {
			columns.push_back(vec4(mattoTranspose[0][i], mattoTranspose[1][i], mattoTranspose[2][i], mattoTranspose[3][i]));
		}
		mat4 transpose(columns[0], columns[1], columns[2], columns[3]);
		matrix = transformInverse * matrix;
		matrix = matrix * transpose;
	}

	vec3 gradf(vec4 r) {

		vec4 grad = r * matrix * 2;
		return vec3(grad.x, grad.y, grad.z);
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec4 dir = { ray.dir.x, ray.dir.y, ray.dir.z, 0 };
		vec4 start = { ray.start.x, ray.start.y, ray.start.z, 1 };
		float a = f(dir);
		float b = dot(start * matrix, dir) + dot(dir * matrix, start);
		float c = f(start);
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		if (cut) {
			for (Plane i : p) {
				if (i.f(ray.start + ray.dir * t1) > 0.0) t1 = -1;
				if (i.f(ray.start + ray.dir * t2) > 0.0) t2 = -1;
			}
		}
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = normalize(gradf(vec4((hit.position - center).x, (hit.position - center).y, (hit.position - center).z, 1)));
		hit.material = material;
		if (dot(hit.normal, ray.dir) > 0) hit.normal = hit.normal * (-1);
		return hit;
	}

	bool isRoom() { return false; }
};

class Elipsoid : public Quadric {
public:
	Elipsoid(vec3 center, float a, float b, float c, Material* _material, std::vector<Plane> p) : Quadric(center, p) {
		material = _material;
		matrix = mat4(1 / (a * a), 0, 0, 0,
			0, 1 / (b * b), 0, 0,
			0, 0, 1 / (c * c), 0,
			0, 0, 0, -1);
	}
	Elipsoid(vec3 center, float a, float b, float c, Material* _material) : Quadric(center) {
		material = _material;
		matrix = mat4(1 / (a * a), 0, 0, 0,
			0, 1 / (b * b), 0, 0,
			0, 0, 1 / (c * c), 0,
			0, 0, 0, -1);
	}
};

class Hyperboloid : public Quadric {
public:
	Hyperboloid(vec3 center, float a, float b, float c, Material* _material, std::vector<Plane> p) : Quadric(center, p) {
		material = _material;
		matrix = mat4(1 / (a * a), 0, 0, 0,
			0, 1 / (b * b), 0, 0,
			0, 0, -1 / (c * c), 0,
			0, 0, 0, -1);
	}
};

class Cylinder : public Quadric {
public:
	Cylinder(vec3 center, float a, Material* _material, std::vector<Plane> p) : Quadric(center, p) {
		material = _material;
		matrix = mat4(1 / (a * a), 0, 0, 0,
			0, 1 / (a * a), 0, 0,
			0, 0, 0, 0,
			0, 0, 0, -1);
	}
};

class Cone : public Quadric {
public:
	Cone(vec3 center, float a, float b, float c, Material* _material, std::vector<Plane> p) : Quadric(center, p) {
		material = _material;
		matrix = mat4(1 / (a * a), 0, 0, 0,
			0, 1 / (b * b), 0, 0,
			0, 0, -1 / (c * c), 0,
			0, 0, 0, 0);
	}
};

struct Sphere : public Intersectable {
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) * (1.0f / radius);
		hit.material = material;
		return hit;
	}
};

class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

float rnd() { return (float)rand() / RAND_MAX; }

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;
	Circle cir;
public:
	void buildWinnieThePooh(Material* yellow, Material* red, Material* black) {
		Quadric* elipsoid = new Elipsoid(vec3(0, 0, 0), 0.08, 0.1, 0.2, yellow);
		elipsoid->translate(vec3(0, -0.5, -0.7));
		objects.push_back(elipsoid);
		Quadric* bearleg = new Elipsoid(vec3(0, 0, 0), 0.08, 0.1, 0.2, yellow);
		bearleg->translate(vec3(0.2, -0.5, -0.7));
		objects.push_back(bearleg);
		Quadric* bearbody = new Elipsoid(vec3(0, 0, 0), 0.25, 0.2, 0.3, yellow);
		bearbody->translate(vec3(0.1, -0.5, -0.3));
		objects.push_back(bearbody);
		Quadric* bearhead = new Elipsoid(vec3(0, 0, 0), 0.25, 0.2, 0.2, yellow);
		bearhead->translate(vec3(0.1, -0.5, 0.18));
		objects.push_back(bearhead);
		Quadric* beareye1 = new Elipsoid(vec3(0, 0, 0), 0.02, 0.02, 0.02, black);
		beareye1->translate(vec3(0.18, -0.17, 0.195));
		objects.push_back(beareye1);
		Quadric* beareye2 = new Elipsoid(vec3(0, 0, 0), 0.02, 0.02, 0.02, black);
		beareye2->translate(vec3(0.03, -0.17, 0.195));
		objects.push_back(beareye2);
		Quadric* bearear1 = new Elipsoid(vec3(0, 0, 0), 0.07, 0.04, 0.11, yellow);
		bearear1->translate(vec3(0.22, -0.5, 0.36));
		objects.push_back(bearear1);
		Quadric* bearear2 = new Elipsoid(vec3(0, 0, 0), 0.07, 0.04, 0.11, yellow);
		bearear2->translate(vec3(-0.02, -0.5, 0.36));
		objects.push_back(bearear2);
		Quadric* beararm1 = new Elipsoid(vec3(0, 0, 0), 0.1, 0.1, 0.2, yellow);
		beararm1->translate(vec3(0.4, -0.5, -0.3));
		objects.push_back(beararm1);
		std::vector<Plane> planesForBeararmshirt; planesForBeararmshirt.push_back(Plane(vec3(0, 0, -0.21), vec3(0, 0, -1)));
		Quadric* beararmshirt1 = new Elipsoid(vec3(0, 0, 0), 0.11, 0.11, 0.2, red, planesForBeararmshirt);
		beararmshirt1->translate(vec3(0.4, -0.5, -0.3));
		objects.push_back(beararmshirt1);
		Quadric* beararm2 = new Elipsoid(vec3(0, 0, 0), 0.1, 0.1, 0.2, yellow);
		beararm2->translate(vec3(-0.2, -0.5, -0.3));
		objects.push_back(beararm2);
		Quadric* beararmshirt2 = new Elipsoid(vec3(0, 0, 0), 0.11, 0.11, 0.2, red, planesForBeararmshirt);
		beararmshirt2->translate(vec3(-0.2, -0.5, -0.3));
		objects.push_back(beararmshirt2);
		Quadric* bearsnout = new Elipsoid(vec3(0, 0, 0), 0.09, 0.09, 0.09, yellow);
		bearsnout->translate(vec3(0.1, -0.22, 0.14));
		objects.push_back(bearsnout);
		std::vector<Plane> planesForBearshirt; planesForBearshirt.push_back(Plane(vec3(0, 0, -0.29), vec3(0, 0, -1)));
		Quadric* bearshirt = new Elipsoid(vec3(0, 0, 0), 0.27, 0.22, 0.3, red, planesForBearshirt);
		bearshirt->translate(vec3(0.1, -0.5, -0.3));
		objects.push_back(bearshirt);
	}

	void build() {
		vec3 eye = vec3(0, 1.8, 0), vup = vec3(0, 0, 1), lookat = vec3(0, 0, 0);
		float fov = 100 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.33f, 0.33f, 0.33f);
		vec3 lightDirection(1, 20, 5), Le(50, 50, 50);
		lights.push_back(new Light(lightDirection, Le));

		vec3 kd(0.7f, 0.3f, 0.3f), ks(2, 2, 2);
		Material* material = new RoughMaterial(kd, ks, 50);
		Material* blue = new RoughMaterial(vec3(0.0, 0.2, 0.3), ks, 50);
		Material* purple = new RoughMaterial(vec3(0.3, 0.1, 0.4), ks, 50);
		Material* red = new RoughMaterial(vec3(0.7, 0.0, 0.0), ks, 50);
		Material* yellow = new RoughMaterial(vec3(1, 0.7, 0), ks, 20);
		Material* black = new RoughMaterial(vec3(0, 0, 0), ks, 20);
		Material* gold = new ReflectiveMaterial(vec3(0.17, 0.35, 1.5), vec3(3.1, 2.7, 1.9));
		Material* silver = new ReflectiveMaterial(vec3(0.14, 0.16, 0.13), vec3(4.1, 2.3, 3.1));

		float spheroid_b = 2, spheroid_c = 1;
		std::vector<Plane> planeforRoom; planeforRoom.push_back(Plane(vec3(0, 0, 0.95), vec3(0, 0, 1)));
		objects.push_back(new Elipsoid(vec3(0, 0, 0), 2, spheroid_b, spheroid_c, material, planeforRoom));
		cir = Circle(vec3(0, 0, 1.06), sqrtf((1 - (0.95 * 0.95) / (spheroid_c * spheroid_c)) * (spheroid_b * spheroid_b)));

		buildWinnieThePooh(yellow, red, black);

		std::vector<Plane> planesForHyperboloid; planesForHyperboloid.push_back(Plane(vec3(0, 0, 0.1), vec3(0, 0, 1))); planesForHyperboloid.push_back(Plane(vec3(0, 0, -1.2), vec3(0, 0, -1)));
		Quadric* h = new Hyperboloid(vec3(0, 0, 0), 0.13, 0.13, 0.3, gold, planesForHyperboloid);
		h->translate(vec3(-0.4, 0.1, -0.3));
		objects.push_back(h);


		std::vector<Plane> planesForCylinder; planesForCylinder.push_back(Plane(vec3(0, 0, 0.3), vec3(0, 0, 1))); planesForCylinder.push_back(Plane(vec3(0, 0, -1.2), vec3(0, 0, -1)));
		Quadric* c = new Cylinder(vec3(0, 0, 0), 0.3, blue, planesForCylinder);
		c->translate(vec3(1.3, -0.7, 0));	 
		objects.push_back(c);


		std::vector<Plane> planesForCone; planesForCone.push_back(Plane(vec3(0, 0, -0.6), vec3(0, 0, 1)));planesForCone.push_back(Plane(vec3(0, 0, -1.5), vec3(0, 0, -1)));
		Quadric* co = new Cone(vec3(0, 0, 0), 0.1, 0.1, 0.2, red, planesForCone);
		co->translate(vec3(0.4, 0.2, -0.6));
		objects.push_back(co);

		std::vector<Plane> planesForSolarPipe; planesForSolarPipe.push_back(Plane(vec3(0, 0, 0.95), vec3(0, 0, -1))); planesForSolarPipe.push_back(Plane(vec3(0, 0, 2.5), vec3(0, 0, 1)));
		Quadric* solarpipe = new Hyperboloid(vec3(0, 0, 0), cir.getRadius(), cir.getRadius(), 1.1, silver, planesForSolarPipe);
		solarpipe->translate(vec3(0, 0, 0.95));
		objects.push_back(solarpipe);
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (int i = 0; i < objects.size() - 1; i++)
			if (objects[i]->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		if (depth > 11)
			return La;

		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return 2 * vec3(0.58, 0.79, 0.98) + lights[0]->Le * pow(dot(ray.dir, lights[0]->direction), 10);
		vec3 outRadiance(0, 0, 0);

		if (hit.material->type == ROUGH) {
			outRadiance = hit.material->ka * La;
			for (vec3 point : cir.getpoints()) {
				Ray rayToCirclePoint = Ray(hit.position + hit.normal * epsilon, point - hit.position);
				float cosTheta = dot(hit.normal, normalize(point - hit.position));
				if (cosTheta > 0 && !shadowIntersect(rayToCirclePoint)) {
					vec3 intensity = trace(rayToCirclePoint);
					float distance = (length(point - hit.position) * length(point - hit.position));
					float deltaOmega = cir.getAreaOfPoint() * dot(normalize(vec3(0, 0, 1)), normalize(point - hit.position)) / distance;
					outRadiance = outRadiance + intensity * hit.material->kd * cosTheta * deltaOmega;
					vec3 halfway = normalize(-ray.dir + normalize(point - hit.position));
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance = outRadiance + intensity * hit.material->ks * powf(cosDelta, hit.material->shininess) * deltaOmega;
				}
			}
		}

		if (hit.material->type == REFLECTIVE) {
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			float cosa = -dot(ray.dir, hit.normal);
			vec3 one(1, 1, 1);
			vec3 F = hit.material->f0 + (one - hit.material->f0) * pow(1 - cosa, 5);
			outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
		}
		return outRadiance;
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;



class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}
