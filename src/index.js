import {get_default_lil_gui, ThreeEngine} from "../js/utils/utils_three.js";
import {
    DoubleSide,
    PCFSoftShadowMap,
    MeshPhysicalMaterial,
    TextureLoader,
    FloatType,
    PMREMGenerator,
    Scene,
    PerspectiveCamera,
    WebGLRenderer,
    Color,
    ACESFilmicToneMapping,
    sRGBEncoding,
    Mesh,
    SphereGeometry,
    MeshBasicMaterial,
    Vector2,
    DirectionalLight,
    Clock,
    RingGeometry,
    Vector3,
    PlaneGeometry,
    CameraHelper,
    Group,
  } from "https://cdn.skypack.dev/three@0.137";
  import { RGBELoader } from "https://cdn.skypack.dev/three-stdlib@2.8.5/loaders/RGBELoader";
  import { OrbitControls } from "https://cdn.skypack.dev/three-stdlib@2.8.5/controls/OrbitControls";
  import { GLTFLoader } from "https://cdn.skypack.dev/three-stdlib@2.8.5/loaders/GLTFLoader";
  import anime from 'https://cdn.skypack.dev/animejs@3.2.1';

// A must be a 3x3 matrix in row major order
// [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]
function matrix_inverse_3x3(A) {
    let det = A[0][0] * (A[1][1] * A[2][2] - A[2][1] * A[1][2]) -
        A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
        A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);

    if (det === 0) {
        return null; // No inverse exists if determinant is 0
    }

    let cofactors = [
        [
            (A[1][1] * A[2][2] - A[2][1] * A[1][2]),
            -(A[1][0] * A[2][2] - A[1][2] * A[2][0]),
            (A[1][0] * A[2][1] - A[2][0] * A[1][1])
        ],
        [
            -(A[0][1] * A[2][2] - A[0][2] * A[2][1]),
            (A[0][0] * A[2][2] - A[0][2] * A[2][0]),
            -(A[0][0] * A[2][1] - A[2][0] * A[0][1])
        ],
        [
            (A[0][1] * A[1][2] - A[0][2] * A[1][1]),
            -(A[0][0] * A[1][2] - A[1][0] * A[0][2]),
            (A[0][0] * A[1][1] - A[1][0] * A[0][1])
        ]
    ];

    let adjugate = [
        [cofactors[0][0] / det, cofactors[1][0] / det, cofactors[2][0] / det],
        [cofactors[0][1] / det, cofactors[1][1] / det, cofactors[2][1] / det],
        [cofactors[0][2] / det, cofactors[1][2] / det, cofactors[2][2] / det]
    ];

    return adjugate;
}

function transpose(matrix) {
    return matrix[0].map((col, i) => matrix.map(row => row[i]));
}

function scalar_mult(mat, k) {
    for (var i = 0; i < 3; i++) {
        for (var j = 0; j < 3; j++) {
            mat[i][j] = mat[i][j] * k;        
        }
    }
    return mat;
}

function add(A, B) {
    let C = new Array(3);
    for (var k = 0; k < 3; k++) {
        C[k] = new Array(3); 
    }

    for (var i = 0; i < 3; i++) {
        for (var j = 0; j < 3; j++) {
            C[i][j] = A[i][j] + B[i][j];  
        }
    }
    return C;
} 

function matvec(A,x) {
    let y = new Array(3);
    for (var i = 0; i < 3; i++) {
        var inner_prod = 0;
        for (var j = 0; j < 3; j++) {
            inner_prod += A[i][j]*x[j];
        }
        y[i] = inner_prod;
    }
    return y;
}

function se3(R,t) {
    let eye = [[1,0,0],[0,1,0],[0,0,1]];

    let beta = Math.acos((R[0][0]+R[1][1]+R[2][2]-1)/2);
    if (beta == Math.PI) {
        var A = [[0,-Math.PI*Math.sqrt(0.5*(R[2][2]+1)),Math.PI*Math.sqrt(0.5*(R[1][1]+1))],[Math.PI*Math.sqrt(0.5*(R[2][2]+1)),0,-Math.PI*Math.sqrt(0.5*(R[0][0]+1))],[-Math.PI*Math.sqrt(0.5*(R[1][1]+1)),Math.PI*Math.sqrt(0.5*(R[0][0]+1)),0]];
    } else if (beta > 0.001) {
        var kA = beta/(2*Math.sin(beta));
        var temp1 = scalar_mult(transpose(R),-1);
        var temp2 = add(R,temp1);
        var A = scalar_mult(temp2,kA);
    } else {
        var kA = 1/2 + (beta**2)/12 + (7*beta**4)/720;
        var temp1 = scalar_mult(transpose(R),-1);
        var temp2 = add(R,temp1);
        var A = scalar_mult(temp2,kA);
    }

    var B = multiplyMatrices(A,A);
    if (beta < 0.001) {
        var p2 = 1/2 + (beta**2)/24 + (beta**4)/720
        var q2 = 1/3 + (beta**2)/120 + (beta**4)/5040
    } else {
        var p2 = (1-Math.cos(beta))/(beta**2);
        var q2 = (beta-Math.sin(beta))/(beta**3);
    }
    let Atemp = scalar_mult(A,p2);
    let Btemp = scalar_mult(B,q2);
    let temp6 = add(Atemp,Btemp);
    let S = add(eye,temp6);
    let Sinv = matrix_inverse_3x3(S);
    let a_arr = matvec(Sinv,t);

    let ret_mat = [[A[0][0],A[0][1],A[0][2],a_arr[0]],[A[1][0],A[1][1],A[1][2],a_arr[1]],[A[2][0],A[2][1],A[2][2],a_arr[2]],[0,0,0,1]];
    return ret_mat;
}

function multiplyMatrices(m1, m2) {
    var result = [];
    for (var i = 0; i < m1.length; i++) {
        result[i] = [];
        for (var j = 0; j < m2[0].length; j++) {
            var sum = 0;
            for (var k = 0; k < m1[0].length; k++) {
                sum += m1[i][k] * m2[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}

function SE3(a,b,c,d,e,f) {
    let eye = [[1,0,0],[0,1,0],[0,0,1]];
    let A = [[0,-c,b],[c,0,-a],[-b,a,0]];
    let B = [[-1*(b**2+c**2),a*b,a*c],[a*b,-1*(a**2+c**2),b*c],[a*c,b*c,-1*(a**2+b**2)]];

    let beta = Math.sqrt(a**2+b**2+c**2);
    if (beta < 0.001) {
        var p1 = 1-(beta**2)/6+(beta**4)/120;
        var q1 = (1/2)-(beta**2)/24+(beta**4)/720;
    } else {
        var p1 = Math.sin(beta)/beta;
        var q1 = (1-Math.cos(beta))/(beta**2);
    }

    let temp1 = scalar_mult(A,p1);
    let temp2 = scalar_mult(B,q1);
    let temp3 = add(temp1,temp2);
    let R = add(eye,temp3);

    if (beta < 0.001) {
        var p2 = (1/2)+(beta**2)/24+(beta**4)/720;
        var q2 = (1/6)+(beta**2)/120+(beta**4)/5040;
    } else {
        var p2 = (1-Math.cos(beta))/(beta**2);
        var q2 = (beta-Math.sin(beta))/(beta**3);
    }
    let Atemp = scalar_mult(A,p2);
    let Btemp = scalar_mult(B,q2);
    let temp6 = add(Atemp,Btemp);
    let S = add(eye,temp6);
    //let t = matvec(S,[d,e,f]);
    let t = [d,e,f];

    let ret_mat = [R,t];
    return ret_mat;
}

function multiply_3x3_matrix_and_3x1_vector(matrix, vector, transl) {
    let return_vec = [];
    for (var i = 0; i < 3; i++) {
        let innerProd = 0;
        for (var j = 0; j < 3; j++) {
            innerProd = innerProd + matrix[i][j]*vector[j][0];
        }
        return_vec.push([innerProd+transl[i]]);
    }
    return return_vec;
}

const scene = new Scene();

let sunBackground = document.querySelector(".sun-background");

const camera = new PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 1000);
camera.position.set(0, 15, 50);

const renderer = new WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = ACESFilmicToneMapping;
renderer.outputEncoding = sRGBEncoding;
renderer.physicallyCorrectLights = true;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

const sunLight = new DirectionalLight(
    new Color("#FFFFFF").convertSRGBToLinear(),
    3.5,
);

sunLight.position.set(10, 20, 10);
sunLight.castShadow = true;
sunLight.shadow.mapSize.width = 512;
sunLight.shadow.mapSize.height = 512;
sunLight.shadow.camera.near = 0.5;
sunLight.shadow.camera.far = 100;
sunLight.shadow.camera.left = -10;
sunLight.shadow.camera.bottom = -10;
sunLight.shadow.camera.top = 10;
sunLight.shadow.camera.right = 10;
scene.add(sunLight);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.dampingFactor = 0.05;
controls.enableDamping = true;

let mousePos = new Vector2(0,0);

(async function () {
  
    let textures = {
        map: await new TextureLoader().loadAsync("assets/earthmap.jpg"),
    };

    let sphere = new Mesh(
        new SphereGeometry(10, 70, 70),
        new MeshPhysicalMaterial({
        map: textures.map,
        sheen: 1,
        sheenRoughness: 0.75,
        clearcoat: 0.5,
        }),
    );
    sphere.receiveShadow = true;
    scene.add(sphere);

    let plane = (await new GLTFLoader().loadAsync("assets/plane/scene.glb")).scene.children[0];
    let planesData = [
        makePlane(plane, scene),
        makePlane(plane, scene),
        makePlane(plane, scene),
    ];

    let clock = new Clock();


    renderer.setAnimationLoop(() => {

    let delta = clock.getDelta();
    sphere.rotation.y += delta * 0.05;

    controls.update();
    renderer.render(scene, camera);


    planesData.forEach(planeData => {
        let plane = planeData.group;

        plane.position.set(0,0,0);
        plane.rotation.set(0,0,0);
        plane.updateMatrixWorld();

        // apply rotation, with yoff altitude along random axis with random thetas, and constrainted to face forward
        planeData.rot += delta * 0.5;
        //console.log(planeData);
        // rotation features, three-js embeds the SO3+shift rotations
        plane.rotateOnAxis(planeData.randomAxis, planeData.randomAxisRot); // random axis rotation
        plane.rotateOnAxis(new Vector3(0, 1, 0), planeData.rot);    // y-axis rotation
        plane.rotateOnAxis(new Vector3(0, 0, 1), planeData.radius); // z-axis
        plane.translateY(planeData.yOff);
        plane.rotateOnAxis(new Vector3(1,0,0), + Math.PI*0.5); // rotate x-axis to ensure faceing forward direction
    });

    renderer.autoClear = false;
    });
})();

function makePlane(planeMesh, scene) {

    let plane = planeMesh.clone();
    plane.scale.set(0.001, 0.001, 0.001);
    plane.position.set(0,0,0);
    plane.rotation.set(0,0,0);
    plane.updateMatrixWorld();

    plane.traverse((object) => {
        if (object instanceof Mesh) {
            object.sunEnvIntensity = 1;
            object.castShadow = true;
            object.receiveShadow = true;
        }
    });

    let group = new Group();
    group.add(plane);

    scene.add(group);

    return {
        group,
        yOff: 10 + Math.random() * 1.0,
        rot: 2*Math.PI,
        radius: Math.random()+Math.PI * 0.05,
        randomAxisRot: Math.random()*2*Math.PI,
        randomAxis: new Vector3(Math.random(), Math.random(), Math.random()).normalize(),
    };
}

window.addEventListener("mousemove", (e) => {
    let x = e.clientX - innerWidth * 0.5; 
    let y = e.clientY - innerHeight * 0.5;

    mousePos.x = x * 0.0001;
    mousePos.y = y * 0.0001;
});



/*engine.animation_loop(() => {
    let m1 = SE3(settings.t1_1,settings.t2_1,settings.t3_1,settings.x1,settings.y1,settings.z1)[0];
    let transl1 = SE3(settings.t1_1,settings.t2_1,settings.t3_1,settings.x1,settings.y1,settings.z1)[1];

    let m2 = SE3(settings.t1_2,settings.t2_2,settings.t3_2,settings.x2,settings.y2,settings.z2)[0];
    let transl2 = SE3(settings.t1_2,settings.t2_2,settings.t3_2,settings.x2,settings.y2,settings.z2)[1];

    let new_standard_points1 = [];
    let new_wireframe_points1 = [];

    let new_standard_points2 = [];
    let new_wireframe_points2 = [];
    
    for (var i = 0; i < standard_points1.length; i++) {
        new_standard_points1.push(multiply_3x3_matrix_and_3x1_vector(m1, standard_points1[i],transl1));
    }

    for (var i = 0; i < wireframe_points1.length; i++) {
        new_wireframe_points1.push(multiply_3x3_matrix_and_3x1_vector(m1, wireframe_points1[i],transl1));
    }

    for (var i = 0; i < standard_points2.length; i++) {
        new_standard_points2.push(multiply_3x3_matrix_and_3x1_vector(m2, standard_points2[i],transl2));
    }

    for (var i = 0; i < wireframe_points2.length; i++) {
        new_wireframe_points2.push(multiply_3x3_matrix_and_3x1_vector(m2, wireframe_points2[i],transl2));
    }

    engine.update_vertex_positions_of_mesh_object(0, new_standard_points1);
    engine.update_vertex_positions_of_mesh_object_wireframe(0, new_wireframe_points1);

    engine.update_vertex_positions_of_mesh_object(1, new_standard_points2);
    engine.update_vertex_positions_of_mesh_object_wireframe(1, new_wireframe_points2);

    engine.mesh_objects[0].visible = settings.mesh_visible;
    engine.set_mesh_object_wireframe_visibility(0, settings.wireframe_visible);

    engine.mesh_objects[1].visible = settings.mesh_visible;
    engine.set_mesh_object_wireframe_visibility(1, settings.wireframe_visible);
});*/