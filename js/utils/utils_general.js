
export function print_var_to_document(v, prefix="") {
    let newDiv = document.createElement("div");
    newDiv.innerHTML = prefix + v;
    document.body.appendChild(newDiv);
}