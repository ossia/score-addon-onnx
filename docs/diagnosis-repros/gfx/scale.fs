/*{ "ISFVSN": "2.0", "INPUTS": [ { "NAME": "inputImage", "TYPE": "image" } ] }*/
void main() { vec4 c = IMG_THIS_PIXEL(inputImage); gl_FragColor = vec4(c.rgb * 0.25, 1.0); }
