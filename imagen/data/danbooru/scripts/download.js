const child_process = require("child_process");

const HOST = "176.9.41.242:873";
const OUT_DIR = "../data/dataset";

for (let i = 0; i < 200; ++i) {
  const bucket = i.toString().padStart("4", "0");
  const args = [
    "--info=progress2",
    "-z",
    `rsync://${HOST}/danbooru2021/original/${bucket}/*`,
    `${OUT_DIR}/${bucket}`,
  ];
  const command = `rsync ${args.join(" ")}`;
  console.log("Running command:");
  console.log(command);
  const r = child_process.spawnSync("rsync", args, { stdio: "inherit" });
  if (r.status !== 0) {
    console.log(`Status code: ${r.status}`);
    throw r.error;
  }
}
