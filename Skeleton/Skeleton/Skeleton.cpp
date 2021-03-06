//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2018. osztol.
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
#include "framework.h"
#include <map>

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char* const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char* const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";


GPUProgram gpuProgram; // vertex and fragment shaders

template <std::size_t numberOfCirclePoints>
class Circle {
	unsigned int vao, vbo[2];
	// felvett kor pontossaga

public:
	void create() {
		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active

		glGenBuffers(2, &vbo[0]);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		vec2 vertices[numberOfCirclePoints];				//kor pontjai
		for (int i = 0; i < numberOfCirclePoints; i++) {
			float fi = i * 2 * M_PI / numberOfCirclePoints;
			vertices[i].x = cosf(fi);
			vertices[i].y = sinf(fi);
			//printf("%d, %d", vertices[numberOfCirclePoints].x, vertices[numberOfCirclePoints].y);
		}

		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * numberOfCirclePoints,  // # bytes
			vertices,	      	// address
			GL_STATIC_DRAW);	// we do not change later
		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed

		float color[3 * numberOfCirclePoints];
		for (int i = 0; i < 3 * numberOfCirclePoints; i++) {
			color[i] = 180.0;
		}
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * numberOfCirclePoints, color, GL_STATIC_DRAW);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	void draw() {
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, 118.0f, 118.0f, 118.0f); // 3 floats

		float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
								  0, 1, 0, 0,    // row-major!
								  0, 0, 1, 0,
								  0, 0, 0, 1 };

		gpuProgram.Use();
		location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

		glBindVertexArray(vao);  // Draw call
		glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, numberOfCirclePoints /*# Elements*/);

	}
};

class Triangle {
	unsigned int vao, vbo;
	vec2 vertexdata[3];
	int noOfVertexes = 0;
	std::vector<std::pair<vec2, float>> circledata;
	std::vector<vec2> pointsOfTriangle;
	int accuracy;
public:
	Triangle(): accuracy(100){}

	void create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), NULL);
	}

	void calculatePointsOfCircles() {
		for (int i = 0; i <= 2; i++) {
			int j;
			if (i == 2)		j = 0;
			else			j = i + 1;
			float a = vertexdata[i].x, b = vertexdata[i].y, c = vertexdata[j].x, d = vertexdata[j].y;
			float x = (b * (c * c + d * d + 1) - d * (a * a + b * b + 1)) / (2 * (b * c - a * d));
			float y = (a * (d * d + c * c + 1) - c * (b * b + a * a + 1)) / (2 * (a * d - b * c));
			float r = sqrtf(x * x + y * y - 1);
			circledata.push_back(std::pair<vec2, float>(vec2(x, y), r));
			
		}
		for(int i = 0; i < 3; i++)
			printf("x: %.2f y: %.2f r: %f\n", circledata[i].first.x, circledata[i].first.y, circledata[i].second);
	}

	void calculatePointsOfTriangle() {
		std::vector<std::vector<vec2>> lineBetweenPoints;
		for (int i = 0; i <= 2; i++) {
			int j;
			if (i == 2)		j = 0;
			else			j = i + 1;
			std::vector<vec2> line;
			for (float k = 1.0/accuracy; k < 1.0; k += (1.0/accuracy)) {
				line.push_back(vertexdata[i] * k + vertexdata[j] * (1 - k));
			}
			//[0] - vertexdata[0] and vertexdata[1], [1] - vertexdata[1] and vertexdata[2], [2] - vertexdata[2] and vertexdata[0]
			lineBetweenPoints.push_back(line);
		}

		for (int i = 0; i < 3; i++) {
			int j;
			if (i == 2)		j = 0;
			else			j = i + 1;
			pointsOfTriangle.push_back(vertexdata[i]);
			std::vector<vec2> pOc;
			for (vec2 k : lineBetweenPoints[i]) {
				vec2 v = k - circledata[i].first;
				v = normalize(v);
				v = v * circledata[i].second;
				v = v + circledata[i].first;
				pOc.push_back(v);
			}
			pointsOfTriangle.insert(pointsOfTriangle.end(), pOc.rbegin(), pOc.rend());
		}
	}

	float calculateLengthOfSides() {
		for (int i = 0; i < accuracy; i++) {

		}
		return NULL;
	}
	void addpoint(float cx, float cy) {
		if (noOfVertexes < 3) {
			vertexdata[noOfVertexes].x = cx;
			printf("%f\t", vertexdata[noOfVertexes].x);
			vertexdata[noOfVertexes].y = cy;
			printf("%f\n", vertexdata[noOfVertexes].y);
			noOfVertexes++;
		}
		if (noOfVertexes == 3) {
			calculatePointsOfCircles();
			calculatePointsOfTriangle();
			calculateLengthOfSides();
		}
	}

	void draw() {
		if (noOfVertexes == 3) {
			int location = glGetUniformLocation(gpuProgram.getId(), "color");
			glUniform3f(location, 0.0f, 1.0f, 0.0f); // 3 floats

			float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
									  0, 1, 0, 0,    // row-major!
									  0, 0, 1, 0,
									  0, 0, 0, 1 };

			gpuProgram.Use();
			location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
			glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

			glBindVertexArray(vao);
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferData(GL_ARRAY_BUFFER, sizeof(vec2) * pointsOfTriangle.size(), &pointsOfTriangle[0], GL_STATIC_DRAW);
			glLineWidth(1.0);
			glDrawArrays(GL_LINE_LOOP, 0, pointsOfTriangle.size());
		}
	}
};

Circle<100> cir;
Triangle t;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	cir.create();
	t.create();

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	cir.draw();
	t.draw();

	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	//printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	if (state == GLUT_DOWN && button == GLUT_LEFT_BUTTON) {
		t.addpoint(cX, cY);
	}
	glutPostRedisplay();
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
