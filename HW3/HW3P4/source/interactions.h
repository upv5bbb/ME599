#ifndef INTERACTIONS_H
#define INTERACTIONS_H
#define W 600
#define H 600
#define TITLE_STRING "Stability"


void keyboard(unsigned char key, int x, int y) {
	if (key == 27) exit(0);
	glutPostRedisplay();
}

//void handleSpecialKeypress(int key, int x, int y) {
//  if (key == GLUT_KEY_DOWN) param -= DELTA_P;
//  if (key == GLUT_KEY_UP) param += DELTA_P;
//  glutPostRedisplay();
//}

// no mouse interactions implemented for this app
void mouseMove(int x, int y) { return; }
void mouseDrag(int x, int y) { return; }

void printInstructions() {
	printf("Newton Iteration visualizer\n");
}

#endif