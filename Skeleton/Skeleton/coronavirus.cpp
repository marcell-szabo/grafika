//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
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
// Nev    : Szabo Marcell
// Neptun : LAUPQQ
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
// Computer Graphics Sample Program: 3D engine-let
// Shader: Gouraud, Phong, NPR
// Material: diffuse + Phong-Blinn
// Texture: CPU-procedural
// Geometry: sphere, torus, mobius
// Camera: perspective
// Light: point
//=============================================================================================
//=============================================================================================
// Computer Graphics Sample Program: 3D engine-let
// Shader: Gouraud, Phong, NPR
// Material: diffuse + Phong-Blinn
// Texture: CPU-procedural
// Geometry: sphere, torus, mobius
// Camera: perspective
// Light: point
//=============================================================================================
#include "framework.h"

//---------------------------
template<class T> struct Dnum { //Dual numbers for automatic derivation
//---------------------------
    float f; //function value
    T d;
    Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
    Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
    Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
    Dnum operator*(Dnum r) {
        return Dnum(f * r.f, f * r.d + d * r.f);
    }
    Dnum operator/(Dnum r) {
        return Dnum(f / r.f, (r.f * d - r.d * r.f) / r.f / r.f);
    }
};

// Elementary functions prepared for the chain rule as well
template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f) * g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T> g) { return Dnum<T>(cos(g.f), -sin(g.f) * g.d); }
template<class T> Dnum<T> Tan(Dnum<T> g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return Dnum<T>(sinhf(g.f), coshf(g.f) * g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return Dnum<T>(coshf(g.f), sinhf(g.f) * g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
    return Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}

typedef Dnum<vec2> Dnum2;

const int tessellationLevel = 30;
//-------------------------- -
struct Camera { // 3D camera
//---------------------------
    vec3 wEye, wLookat, wVup;   // extrinsic
    float fov, asp, fp, bp;		// intrinsic
public:
    Camera() {
        asp = (float)windowWidth / windowHeight;
        fov = 110.0f * (float)M_PI / 180.0f;
        fp = 1; bp = 20;
    }
    mat4 V() { // view matrix: translates the center to the origin
        vec3 w = normalize(wEye - wLookat);
        vec3 u = normalize(cross(wVup, w));
        vec3 v = cross(w, u);
        return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
            u.y, v.y, w.y, 0,
            u.z, v.z, w.z, 0,
            0, 0, 0, 1);
    }

    mat4 P() { // projection matrix
        return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
            0, 1 / tan(fov / 2), 0, 0,
            0, 0, -(fp + bp) / (bp - fp), -1,
            0, 0, -2 * fp * bp / (bp - fp), 0);
    }
};

//---------------------------
struct Material {
    //---------------------------
    vec3 kd, ks, ka;
    float shininess;
};

//---------------------------
struct Light {
    //---------------------------
    vec3 La, Le;
    vec4 wLightPos;
};

//---------------------------
class CheckerBoardTexture : public Texture {
    //---------------------------
public:
    CheckerBoardTexture(const int width = 0, const int height = 0) : Texture() {
        std::vector<vec4> image(width * height);
        const vec4 yellow(0.5, 1, 0.7, 1), blue(0.1, 0.5, 1, 1);
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++) {
                image[y * width + x] = (y & 1) ? yellow : blue;
            }
        create(width, height, image, GL_NEAREST);
    }
};

class SpikeTexture : public Texture {
public:
    SpikeTexture(const int width = 0, const int height = 0) : Texture() {
        std::vector<vec4> image(width * height);
        const vec4 red(1, 0, 0, 1), purple(0.3, 0, 0.7, 1);
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
                image[y * width + x] = (x < (width / 2)) ? red : purple;
        create(width, height, image, GL_NEAREST);
    }
};

//---------------------------
struct RenderState {
    //---------------------------
    mat4	            MVP, M, Minv, V, P;
    Material* material;
    std::vector<Light>  lights;
    Texture* texture;
    vec3	            wEye;
};

//---------------------------
class Shader : public GPUProgram {
    //---------------------------
public:
    virtual void Bind(RenderState state) = 0;

    void setUniformMaterial(const Material& material, const std::string& name) {
        setUniform(material.kd, name + ".kd");
        setUniform(material.ks, name + ".ks");
        setUniform(material.ka, name + ".ka");
        setUniform(material.shininess, name + ".shininess");
    }

    void setUniformLight(const Light& light, const std::string& name) {
        setUniform(light.La, name + ".La");
        setUniform(light.Le, name + ".Le");
        setUniform(light.wLightPos, name + ".wLightPos");
    }
};

//---------------------------
class PhongShader : public Shader {
    //---------------------------
    const char* vertexSource = R"(
		#version 330
		precision highp float;
 
		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};
 
		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye
 
		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;
 
		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec2 texcoord;
 
		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

    // fragment shader in GLSL
    const char* fragmentSource = R"(
		#version 330
		precision highp float;
 
		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};
 
		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};
 
		uniform Material material;
		uniform Light[8] lights;    // light sources
		uniform int   nLights;
		uniform sampler2D diffuseTexture;
 
		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		in  vec2 texcoord;
 
        out vec4 fragmentColor; // output goes to frame buffer
 
		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView);
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;
 
			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La +
                           (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
    PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

    void Bind(RenderState state) {
        Use(); 		// make this program run
        setUniform(state.MVP, "MVP");
        setUniform(state.M, "M");
        setUniform(state.Minv, "Minv");
        setUniform(state.wEye, "wEye");
        setUniform(*state.texture, std::string("diffuseTexture"));
        setUniformMaterial(*state.material, "material");

        setUniform((int)state.lights.size(), "nLights");
        for (unsigned int i = 0; i < state.lights.size(); i++) {
            setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
        }
    }
};

//---------------------------
class Geometry {
    //---------------------------
protected:
    unsigned int vao, vbo;        // vertex array object
public:
    Geometry() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
    }
    virtual void Draw() = 0;
    virtual void Animate(float tstart, float tend) {}
    ~Geometry() {
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
    }
};

struct VertexData {
    vec3 position, normal;
    vec2 texcoord;
};

struct Triangle {
    VertexData vtx;
    vec3 vertexes[4];
    float side;

    Triangle(vec3 a, vec3 b, vec3 c) {
        vertexes[0] = a; vertexes[1] = b; vertexes[2] = c;
        side = length(vec3(b - a));
    }

    vec3 getCentre() {
        return vec3((vertexes[0].x + vertexes[1].x + vertexes[2].x) / 3, (vertexes[0].y + vertexes[1].y + vertexes[2].y) / 3, (vertexes[0].z + vertexes[1].z + vertexes[2].z) / 3);
    }

    void calculate() {
        vec3 h_to_d = -cross(normalize(vertexes[0] - getCentre()), normalize(vertexes[1] - getCentre())) * sqrtf(2.0 / 3.0) * side;
        vertexes[3] = getCentre() + h_to_d;

    }
    void calculate(float time) {
        vec3 h_to_d = -cross(normalize(vertexes[0] - getCentre()), normalize(vertexes[1] - getCentre())) * sqrtf(2.0 / 3.0) * side;
        vertexes[3] = getCentre() + h_to_d;
        vertexes[3] = vertexes[3] * (1 + fabs(2 * cosf(time) * sinf(time)));
    }

    void getVertexData(std::vector<VertexData>& v) {
        for (int i = 0; i < 3; i++) {
            VertexData vtx;
            int h, j;
            if (i == 0) { h = 2; j = 1; }
            else if (i == 2) { h = 1; j = 0; }
            else { h = i - 1; j = i + 1; }
            vec3 normal = cross(normalize(vertexes[h] - vertexes[i]), normalize(vertexes[j] - vertexes[i]));
            vtx.position = vertexes[i];
            vtx.normal = normal;
            vtx.texcoord = vec2(0, 0);
            v.push_back(vtx);
        }
    }
};

class Tetrahedron : public Geometry {
    std::vector<VertexData> vertexData;
    Triangle t;
public:
    Tetrahedron(Triangle t) : t(t) { create(t, 0); addData(); }
    Triangle getHalfPoints(Triangle t) {
        return Triangle(vec3((t.vertexes[0] + t.vertexes[1]) * (1.0 / 2.0)), vec3((t.vertexes[1] + t.vertexes[2]) * (1.0 / 2.0)), vec3((t.vertexes[2] + t.vertexes[0]) * (1.0 / 2.0)));
    }
    void create(Triangle t, float time, int depth = 3) {
        if (depth == 0)
            return;
        if (depth == 3)
            t.calculate();
        else
            t.calculate(time);

        int triangleVertexIndexes[12] = { 0,1,2,1,3,0,0,3,2,1,2,3 };
        for (int i = 0; i < 12; i += 3) {
            Triangle(t.vertexes[triangleVertexIndexes[i]], t.vertexes[triangleVertexIndexes[i + 1]], t.vertexes[triangleVertexIndexes[i + 2]]).getVertexData(vertexData);
            if (i == 0) {
                vec3 a = vec3((t.vertexes[triangleVertexIndexes[i]] + t.vertexes[triangleVertexIndexes[i + 1]]) * (1.0 / 2.0)), b = vec3((t.vertexes[triangleVertexIndexes[i + 1]] + t.vertexes[triangleVertexIndexes[i + 2]]) * (1.0 / 2.0)), c = vec3((t.vertexes[triangleVertexIndexes[i + 2]] + t.vertexes[triangleVertexIndexes[i]]) * (1.0 / 2.0));
                create(Triangle(c, b, a), time, depth - 1);
                if (depth - 2 >= 1) {
                    create(getHalfPoints(Triangle(c, a, t.vertexes[triangleVertexIndexes[i]])), time, depth - 2);
                    create(getHalfPoints(Triangle(b, t.vertexes[triangleVertexIndexes[i + 1]], a)), time, depth - 2);
                    create(getHalfPoints(Triangle(t.vertexes[triangleVertexIndexes[i + 2]], b, c)), time, depth - 2);
                }
            }
            else {
                vec3 a = vec3((t.vertexes[triangleVertexIndexes[i]] + t.vertexes[triangleVertexIndexes[i + 1]]) * (1.0 / 2.0)), b = vec3((t.vertexes[triangleVertexIndexes[i + 1]] + t.vertexes[triangleVertexIndexes[i + 2]]) * (1.0 / 2.0)), c = vec3((t.vertexes[triangleVertexIndexes[i + 2]] + t.vertexes[triangleVertexIndexes[i]]) * (1.0 / 2.0));
                create(Triangle(a, b, c), time, depth - 1);
                if (depth - 2 >= 1) {
                    create(getHalfPoints(Triangle(t.vertexes[triangleVertexIndexes[i]], a, c)), time, depth - 2);
                    create(getHalfPoints(Triangle(a, t.vertexes[triangleVertexIndexes[i + 1]], b)), time, depth - 2);
                    create(getHalfPoints(Triangle(c, b, t.vertexes[triangleVertexIndexes[i + 2]])), time, depth - 2);
                }
            }
        }
    }
    void addData() {
        glBindVertexArray(vao);
        glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(VertexData), &vertexData[0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
        glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
        glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
        // attribute array, components/attribute, component type, normalize?, stride, offset
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
    }
    // Inherited via Geometry
    void Draw() {
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, vertexData.size());
        glBindVertexArray(0);
    }


    // Inherited via Geometry
    void Animate(float tstart, float tend) override {

        vertexData.clear();
        create(t, tend);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertexData.size() * sizeof(VertexData), &vertexData[0]);

    }
};

//---------------------------
class ParamSurface : public Geometry {
    //---------------------------
    std::vector<VertexData> vertexData;
    VertexData vtxData;
    unsigned int nVtxPerStrip, nStrips;
public:
    ParamSurface() { nVtxPerStrip = nStrips = 0; }

    virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z, float time) = 0;

    virtual VertexData GenVertexData(float u, float v, float time) {

        vtxData.texcoord = vec2(u, v);
        Dnum2 X, Y, Z;
        Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
        eval(U, V, X, Y, Z, time);
        vtxData.position = vec3(X.f, Y.f, Z.f);
        vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
        vtxData.normal = cross(drdU, drdV);
        return vtxData;
    }

    void create(float time, int N = tessellationLevel, int M = tessellationLevel) {
        nVtxPerStrip = (M + 1) * 2;
        nStrips = N;
        // vertices on the CPU
        for (int i = 0; i < N; i++) {
            for (int j = 0; j <= M; j++) {
                vertexData.push_back(GenVertexData((float)j / M, (float)i / N, time));
                vertexData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N, time));
            }
        }
    }
    void addData() {
        glBindVertexArray(vao);
        glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vertexData[0], GL_STATIC_DRAW);
        // Enable the vertex attribute arrays
        glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
        glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
        glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
        // attribute array, components/attribute, component type, normalize?, stride, offset
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
    }

    void Draw() {
        glBindVertexArray(vao);
        for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
        glBindVertexArray(0);
    }

    void Animate(float tstart, float tend) {
        vertexData.clear();
        create(tend);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, nVtxPerStrip * nStrips * sizeof(VertexData), &vertexData[0]);
    }

};

//---------------------------
class Sphere : public ParamSurface {
    //---------------------------
public:
    Sphere(float time) { create(time); addData(); }
    void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z, float time) {
        Dnum2 r = (Cos((U + V) * 6.0f * (float)M_PI) /* *sin(time)*/) * 2.0 + 2.0;
        U = U * 2.0f * (float)M_PI, V = V * (float)M_PI;
        X = r * Cos(U) * Sin(V); Y = r * Sin(U) * Sin(V); Z = r * Cos(V);
    }

};

class RealSphere : public Sphere {
public:
    RealSphere(float time) {create(time); addData();}
    void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z, float time) {        
        U = U * 2.0f * (float)M_PI, V = V * (float)M_PI;
        X =  Cos(U) * Sin(V); Y = Sin(U) * Sin(V); Z = Cos(V);
    }
};

//---------------------------
class Tractricoid : public ParamSurface {
    //---------------------------
public:
    Tractricoid(float time) { create(time); addData(); }
    void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z, float time) {
        const float height = 3.0f;
        U = U * height, V = V * 2 * M_PI;
        X = Cos(V) / Cosh(U); Y = Sin(V) / Cosh(U); Z = U - Tanh(U);
    }
};

//---------------------------
struct Object {
    //---------------------------
    Shader* shader;
    Material* material;
    Texture* texture;
    Geometry* geometry;
    vec3 scale, translation, rotationAxis;
    float rotationAngle;
    mat4 M, Minv;
public:
    Object(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry) :
        scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
        shader = _shader;
        texture = _texture;
        material = _material;
        geometry = _geometry;
    }
    virtual void SetModelingTransform(mat4& M, mat4& Minv) {
        M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
        Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
    }

    mat4 getM() { return M; }
    mat4 getMinv() { return Minv; }

    void Draw(RenderState state) {
        SetModelingTransform(M, Minv);
        state.M = M;
        state.Minv = Minv;
        state.MVP = state.M * state.V * state.P;
        state.material = material;
        state.texture = texture;
        shader->Bind(state);
        geometry->Draw();
    }

    virtual void Animate(float tstart, float tend) {
        rotationAngle = 0.8f * tend;
        geometry->Animate(tstart, tend);
    }
};

class Coronabody : public Object {
public:
    Coronabody(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry) : Object(_shader, _material, _texture, _geometry) {}
};

class Spikes : public Coronabody {
public:
    Spikes(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry) : Coronabody(_shader, _material, _texture, _geometry) {}
    void SetModelingTransform(mat4& M, mat4& Minv) {
        M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
        Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
    }
};

class Coronavirus {
    Sphere* body;
    Coronabody* bodyobj;
    std::vector<Object*> spikes;
    static int const noOfSpikesOnEquator = 35;
    Texture* tspikes;
    Material* mspikes;
public:
    Coronavirus(Shader* s, Material* mbody, Texture* tbody, Material* mspikes, Texture* tspikes) : tspikes(tspikes), mspikes(mspikes) {
        body = new Sphere(1.0);
        bodyobj = new Coronabody(s, mbody, tbody, body);
    }

    void create() {
        float const scaleidxOfBody = 0.7;
        bodyobj->scale = vec3(scaleidxOfBody, scaleidxOfBody, scaleidxOfBody);
        for (float i = 0.0f; i < 1.1; i += 0.125f) {
            int noOfSpikes = noOfSpikesOnEquator * sinf(i * M_PI);
            float u = 0.0f;
            for (int j = 0; j < noOfSpikes; j++) {
                VertexData spikedata = body->GenVertexData(u, i, 0.0);
                Object* spike = new Spikes(bodyobj->shader, mspikes, tspikes, new Tractricoid(0.0));
                spike->scale = vec3(0.3, 0.3, 0.3);
                spike->rotationAxis = cross(vec3(0, 0, 1), spikedata.normal);
                spike->rotationAngle = acosf(dot(normalize(vec3(0, 0, 1)), normalize(spikedata.normal)));


                spike->translation = vec3(scaleidxOfBody * spikedata.position) + vec3(-0.3 * normalize(spikedata.normal));

                spikes.push_back(spike);
                u += 1.0 / (float)noOfSpikes;
            }
        }
    }

    void draw(RenderState state) {
        bodyobj->Draw(state);
        for (Object* i : spikes)
            i->Draw(state);
    }
    void animate(float tstart, float tend) {
        //bodyobj->Animate(tstart, tend);
    }
};

//---------------------------
class Scene {
    //---------------------------
    std::vector<Object*> objects;
    Camera camera; // 3D camera
    std::vector<Light> lights;
    Coronavirus* virus;
    Object* room;
    vec3 brownian;
    bool brownset = false;
public:
    void setBrownianMotion(vec3 dir = vec3(0, 0, 0)) {

        if (dir.x == 0 && dir.y == 0 && dir.z == 0) {
            int random = rand() % 6;
            vec3 dirs[6] = { vec3(1,0,0), vec3(-1,0,0), vec3(0,1,0), vec3(0,-1,0), vec3(0,0,1), vec3(0,0,-1) };
            brownian = dirs[random];
        }
        else {
            int random = rand() % 10;
            if (dir.x == 1) {
                vec3 dirs[10] = { vec3(1,0,0), vec3(1,0,0),vec3(1,0,0),vec3(1,0,0),vec3(1,0,0), vec3(-1,0,0), vec3(0,1,0), vec3(0,-1,0), vec3(0,0,1), vec3(0,0,-1) };
                brownian = dirs[random];
            }
            else if (dir.x == -1) {
                vec3 dirs[10] = { vec3(1,0,0), vec3(-1,0,0),vec3(-1,0,0),vec3(-1,0,0),vec3(-1,0,0), vec3(-1,0,0), vec3(0,1,0), vec3(0,-1,0), vec3(0,0,1), vec3(0,0,-1) };
                brownian = dirs[random];
            }
            else if (dir.y == 1) {
                vec3 dirs[10] = { vec3(1,0,0), vec3(-1,0,0),vec3(0,1,0),vec3(0,1,0),vec3(0,1,0), vec3(0,1,0), vec3(0,1,0), vec3(0,-1,0), vec3(0,0,1), vec3(0,0,-1) };
                brownian = dirs[random];
            }
            else if (dir.y == -1) {
                vec3 dirs[10] = { vec3(1,0,0), vec3(-1,0,0),vec3(0,1,0),vec3(0,-1,0),vec3(0,-1,0), vec3(0,-1,0), vec3(0,-1,0), vec3(0,-1,0), vec3(0,0,1), vec3(0,0,-1) };
                brownian = dirs[random];
            }
            else if (dir.z == 1) {
                vec3 dirs[10] = { vec3(1,0,0), vec3(-1,0,0),vec3(0,1,0),vec3(0,-1,0),vec3(0,0,1), vec3(0,0,1), vec3(0,0,1), vec3(0,0,1), vec3(0,0,1), vec3(0,0,-1) };
                brownian = dirs[random];
            }
            else if (dir.z == -1) {
                vec3 dirs[10] = { vec3(1,0,0), vec3(0,1,0), vec3(0,-1,0), vec3(0,0,1), vec3(0,0,-1), vec3(0,0,-1),vec3(0,0,-1),vec3(0,0,-1),vec3(0,0,-1), vec3(0,0,-1) };
                brownian = dirs[random];
            }
        }
        brownset = true;
    }

    void Build() {
        // Shaders
        Shader* phongShader = new PhongShader();

        // Materials
        Material* material0 = new Material;
        material0->kd = vec3(0.6f, 0.4f, 0.2f);
        material0->ks = vec3(4, 4, 4);
        material0->ka = vec3(0.1f, 0.1f, 0.1f);
        material0->shininess = 100;

        Material* material1 = new Material;
        material1->kd = vec3(0.8f, 0.2f, 0.5f);
        material1->ks = vec3(0.8f, 0.2f, 0.5f);
        material1->ka = vec3(0.8f, 0.2f, 0.5f);
        material1->shininess = 30;

        // Textures
        Texture* texture4x8 = new CheckerBoardTexture(14, 20);
        Texture* spiketexture = new SpikeTexture(4, 8);

        virus = new Coronavirus(phongShader, material0, texture4x8, material1, spiketexture);
        virus->create();
        Triangle t = Triangle(vec3(-1.0f / 2.0f, -sqrtf(3.0f) / 4.0f, 0), vec3(0, sqrtf(3) / 4, 0), vec3(1.0f / 2.0f, -sqrtf(3.0f) / 4.0f, 0));
        Object* antibody = new Object(phongShader, material0, texture4x8, new Tetrahedron(t));
        antibody->scale = vec3(2, 2, 2);
        antibody->rotationAxis = vec3(0, 1, 0);
        antibody->translation = vec3(5, 0, 0);
        objects.push_back(antibody);

        Sphere * roomsphere = new RealSphere(0.0);
        room = new Object(phongShader, material0, texture4x8, roomsphere);
        room->scale = vec3(4, 4, 4);


        // Camera
        camera.wEye = vec3(0, 0, 8);
        camera.wLookat = vec3(0, 0, 0);
        camera.wVup = vec3(0, 1, 0);

        // Lights
        lights.resize(3);
        lights[0].wLightPos = vec4(5, 5, 4, 1);	// ideal point -> directional light source
        lights[0].La = vec3(0.1f, 0.1f, 1);
        lights[0].Le = vec3(3, 0, 0);

        lights[1].wLightPos = vec4(5, 10, 20, 1);	// ideal point -> directional light source
        lights[1].La = vec3(0.2f, 0.2f, 0.2f);
        lights[1].Le = vec3(0, 3, 0);

        lights[2].wLightPos = vec4(-5, 5, 5, 1);	// ideal point -> directional light source
        lights[2].La = vec3(0.1f, 0.1f, 0.1f);
        lights[2].Le = vec3(0, 0, 3);
    }

    void Render() {
        RenderState state;
        state.wEye = camera.wEye;
        state.V = camera.V();
        state.P = camera.P();
        state.lights = lights;
        virus->draw(state);
        room->Draw(state);
        for (Object* obj : objects) obj->Draw(state);
    }

    void Animate(float tstart, float tend) {
        virus->animate(tstart, tend);

        for (Object* obj : objects) {
            if (!brownset) {
                setBrownianMotion();
                brownset = false;
            }
            obj->translation = obj->translation + 0.05 * brownian;
            brownset = false;
            obj->Animate(tstart, tend);
        }
    }
};

Scene scene;

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    scene.Build();
}

// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							// background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
    scene.Render();
    glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    if (key == 'x')
        scene.setBrownianMotion(vec3(1, 0, 0));
    else if (key == 'X')
        scene.setBrownianMotion(vec3(-1, 0, 0));
    else if (key == 'y')
        scene.setBrownianMotion(vec3(0, 1, 0));
    else if (key == 'Y')
        scene.setBrownianMotion(vec3(0, -1, 0));
    else if (key == 'z')
        scene.setBrownianMotion(vec3(0, 0, 1));
    else if (key == 'Z')
        scene.setBrownianMotion(vec3(0, 0, -1));
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { }

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { }

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    static float tend = 0;
    const float dt = 0.1f; // dt is ?infinitesimal?
    float tstart = tend;
    tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

    for (float t = tstart; t < tend; t += dt) {
        float Dt = fmin(dt, tend - t);
        scene.Animate(t, t + Dt);
    }
    glutPostRedisplay();
}