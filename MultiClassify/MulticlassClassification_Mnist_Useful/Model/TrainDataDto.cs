using System;
using Jc.Core;

namespace Jc.Nice.Dto
{
    /// <summary>
    /// TrainData Dto
    /// </summary>
    [Table(Name = "a_TrainData")]
    public class TrainDataDto
    {
        #region Properties
        /// <summary>
        /// Id
        /// </summary>
        [PkField]
        public int Id { get; set; }

        /// <summary>
        /// 目录名称
        /// </summary>
        public string DirName { get; set; }

        /// <summary>
        /// 名称
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Count
        /// </summary>
        public long Count { get; set; }

        /// <summary>
        /// 添加日期
        /// </summary>
        public DateTime? AddDate { get; set; }
        #endregion
    }
}
